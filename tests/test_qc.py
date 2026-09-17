import numpy as np

from cdts.qc import qc_modis_summary, qc_modis_state, qc_sentinel2_scl


def test_qc_modis_summary_levels():
    qa = np.array([0, 1, 2, 3, 99])
    w = qc_modis_summary(qa, wmin=0.2, wmid=0.5, wmax=1.0)
    # good, marginal, snow/ice, cloudy, fill/unknown
    np.testing.assert_allclose(w, [1.0, 0.5, 0.2, 0.2, 0.0])


def test_qc_modis_summary_custom_levels():
    qa = np.array([0, 1, 2])
    w = qc_modis_summary(qa, wmin=0.1, wmid=0.4, wmax=0.9)
    np.testing.assert_allclose(w, [0.9, 0.4, 0.1])


def test_qc_modis_state_good_pixel_is_clear_no_snow_low_aerosol():
    # cloud state bits 0-1 = 00 (clear), aerosol bits 6-7 = 01 (low), no snow bit 12
    qa = np.array([1 << 6], dtype=np.int64)
    w = qc_modis_state(qa, wmin=0.2, wmid=0.5, wmax=1.0)
    np.testing.assert_allclose(w, [1.0])


def test_qc_modis_state_snow_pixel_is_bad():
    # snow bit (bit 12) set
    qa = np.array([1 << 12], dtype=np.int64)
    w = qc_modis_state(qa, wmin=0.2, wmid=0.5, wmax=1.0)
    np.testing.assert_allclose(w, [0.2])


def test_qc_modis_state_cloudy_pixel_is_bad():
    # cloud state bits 0-1 = 01 (cloudy)
    qa = np.array([0b01], dtype=np.int64)
    w = qc_modis_state(qa, wmin=0.2, wmid=0.5, wmax=1.0)
    np.testing.assert_allclose(w, [0.2])


def test_qc_modis_state_high_aerosol_is_bad():
    # aerosol bits 6-7 = 11 (high), clear cloud state
    qa = np.array([0b11 << 6], dtype=np.int64)
    w = qc_modis_state(qa, wmin=0.2, wmid=0.5, wmax=1.0)
    np.testing.assert_allclose(w, [0.2])


def test_qc_sentinel2_scl_classes():
    # 4=vegetation(good), 8=cloud medium prob(mid), 3=cloud shadow(bad), 11=snow(bad)
    scl = np.array([4, 8, 3, 11, 6])
    w = qc_sentinel2_scl(scl, wmin=0.2, wmid=0.5, wmax=1.0)
    np.testing.assert_allclose(w, [1.0, 0.5, 0.2, 0.2, 1.0])
