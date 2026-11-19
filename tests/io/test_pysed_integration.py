"""MDtrace policy and DSF integration with the public pySED reader."""

from types import SimpleNamespace

import numpy as np
import pytest
from netCDF4 import Dataset

from mdtrace.dsf import compute_dsf
from mdtrace.io import NetCDFReader, POSITIONS, VELOCITIES
from mdtrace.io.prepare import prepare_trajectory
from tests.io.fixtures import GPUMD_TEXT, LAMMPS_TEXT


def params(path):
    return SimpleNamespace(
        action='compute', experiment='number', trajectory_path=str(path),
        dsf_qpoints=[0, 0, 0], dsf_q_coordinates='cartesian', num_blocks=1,
        time_step=1., output_data_stride=1, dsf_currents=False,
    )


@pytest.mark.parametrize('text,name', [(GPUMD_TEXT, 'dump.xyz'), (LAMMPS_TEXT, 'dump.data')])
def test_shared_reader_preserves_cache_name_units_and_reuse(tmp_path, text, name):
    from pySED.io import NetCDFReader as PublicReader
    assert NetCDFReader is PublicReader
    source = tmp_path / name
    source.write_text(text)
    cache, _ = prepare_trajectory(source, storage_dtype='float64', compression_level=1)
    assert cache.name == name + '.mdtrace.nc'
    with NetCDFReader(cache) as reader:
        reader.require_current_cache(validate_headers=True)
        assert reader.read(POSITIONS).shape == (2, 2, 3)
        np.testing.assert_allclose(reader.read(VELOCITIES)[0, 0], [100, 200, 300])
    with Dataset(cache) as saved:
        assert saved.program == 'pySED'
        assert saved.variables['coordinates'].filters()['zlib']
        assert saved.pysed_headers_validated == 1
    stamp = cache.stat().st_mtime_ns
    prepare_trajectory(source, storage_dtype='float64', compression_level=1)
    assert cache.stat().st_mtime_ns == stamp
    result = compute_dsf(params(cache))
    np.testing.assert_allclose(result['sqw'][0], [0, .004])


def test_dsf_rejects_actual_variable_cell_after_shared_conversion(tmp_path):
    source = tmp_path / 'variable.xyz'
    lines = GPUMD_TEXT.splitlines()
    lines[5] = lines[5].replace('10 0 0', '11 0 0', 1)
    source.write_text('\n'.join(lines) + '\n')
    cache, _ = prepare_trajectory(source)
    with pytest.raises(ValueError, match='fixed simulation cell'):
        compute_dsf(params(cache))


def test_dsf_rebuilds_unchecked_pysed_cache_from_text(tmp_path):
    from pySED.io import prepare_trajectory as prepare_sed
    source = tmp_path / 'dump.xyz'
    source.write_text(GPUMD_TEXT)
    cache, _ = prepare_sed(source)
    with pytest.raises(ValueError, match='header validation'):
        compute_dsf(params(cache))
    prepare_trajectory(source, converted_path=cache)
    assert np.isfinite(compute_dsf(params(cache))['sqw']).all()


def test_legacy_mdtrace_cache_rebuilt_at_same_path_without_editing_source(tmp_path):
    source = tmp_path / 'dump.xyz'
    source.write_text(GPUMD_TEXT)
    cache, _ = prepare_trajectory(source)
    with Dataset(cache, 'a') as saved:
        for key in list(saved.ncattrs()):
            if key.startswith('pysed_'):
                saved.delncattr(key)
        saved.mdtrace_schema_version = 3
        saved.mdtrace_conversion_complete = 1
    before = source.read_bytes()
    prepare_trajectory(source)
    with Dataset(cache) as saved:
        assert saved.program == 'pySED'
        assert saved.pysed_headers_validated == 1
    assert source.read_bytes() == before


def test_dsf_does_not_expose_sed_direct_text_block_reader(tmp_path):
    with pytest.raises(ValueError, match='trajectory_read_mode = cache'):
        prepare_trajectory(tmp_path / 'unused.xyz', text_mode='direct')
