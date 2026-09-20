import numpy as np
import pytest

from ls_bayesian.mcmc.output import (
    AcceptanceQoI,
    BatchMeanStatistic,
    ComponentQoI,
    IdentityStatistic,
    MCMCOutput,
    MeanQoI,
    RunningMeanStatistic,
    build,
)

pytestmark = pytest.mark.unit


# ==================================================================================================
def test_component_qoi_extracts_indexed_value() -> None:
    state = np.array([1.0, 2.0, 3.0])

    assert ComponentQoI(1).evaluate(state, accepted=True) == 2.0
    assert ComponentQoI(-1).evaluate(state, accepted=True) == 3.0


# --------------------------------------------------------------------------------------------------
def test_mean_qoi_matches_np_mean() -> None:
    state = np.array([1.0, 2.0, 3.0, 4.0])

    np.testing.assert_allclose(MeanQoI.evaluate(state, accepted=True), np.mean(state))


# --------------------------------------------------------------------------------------------------
def test_acceptance_qoi_reports_flag_independent_of_state() -> None:
    state = np.array([123.0])

    assert AcceptanceQoI.evaluate(state, accepted=True) == 1.0
    assert AcceptanceQoI.evaluate(state, accepted=False) == 0.0


# ==================================================================================================
def test_running_mean_statistic_matches_incremental_np_mean() -> None:
    values = [3.0, 7.0, -2.0, 5.5, 0.0]
    statistic_under_test = RunningMeanStatistic()

    for step, value in enumerate(values, start=1):
        running_value = statistic_under_test.evaluate(value)
        np.testing.assert_allclose(running_value, np.mean(values[:step]))


# ==================================================================================================
def test_batch_mean_statistic_holds_value_until_batch_completes() -> None:
    statistic_under_test = BatchMeanStatistic(batch_size=3)
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

    results = [statistic_under_test.evaluate(value) for value in values]

    np.testing.assert_allclose(
        results,
        [0.0, 0.0, 2.0, 2.0, 2.0, 5.0, 5.0],
    )


# --------------------------------------------------------------------------------------------------
def test_batch_mean_statistic_batch_size_one_updates_every_value() -> None:
    statistic_under_test = BatchMeanStatistic(batch_size=1)

    results = [statistic_under_test.evaluate(value) for value in [1.0, 2.0, 3.0]]

    np.testing.assert_allclose(results, [1.0, 2.0, 3.0])


# --------------------------------------------------------------------------------------------------
def test_batch_mean_statistic_rejects_non_positive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        BatchMeanStatistic(batch_size=0)


# ==================================================================================================
def test_mcmc_output_records_statistic_value_not_raw_qoi() -> None:
    output_under_test = MCMCOutput(ComponentQoI(0), RunningMeanStatistic())

    output_under_test.update(np.array([2.0]), accepted=True)
    output_under_test.update(np.array([4.0]), accepted=True)

    # raw QoI values would be [2.0, 4.0]; the running mean is [2.0, 3.0].
    np.testing.assert_allclose(output_under_test.all_values, [2.0, 3.0])
    assert output_under_test.value == 3.0


# --------------------------------------------------------------------------------------------------
def test_mcmc_output_all_values_does_not_alias_internal_state() -> None:
    output_under_test = MCMCOutput(ComponentQoI(0), IdentityStatistic())
    output_under_test.update(np.array([1.0]), accepted=True)

    first_read = output_under_test.all_values
    first_read[0] = 999.0

    assert output_under_test.value == 1.0
    np.testing.assert_allclose(output_under_test.all_values, [1.0])


# ==================================================================================================
def test_mcmc_output_rejects_log_true_without_str_id() -> None:
    with pytest.raises(ValueError, match="str_id"):
        MCMCOutput(ComponentQoI(0), IdentityStatistic(), str_format="<+12.3e", log=True)


# --------------------------------------------------------------------------------------------------
def test_mcmc_output_rejects_log_true_without_str_format() -> None:
    with pytest.raises(ValueError, match="str_format"):
        MCMCOutput(ComponentQoI(0), IdentityStatistic(), str_id="foo", log=True)


# ==================================================================================================
def test_build_str_id_uses_qoi_name_alone_for_identity_statistic() -> None:
    output_under_test = build(AcceptanceQoI(), IdentityStatistic())

    assert output_under_test.str_id == f"{'acceptance':<12}"


# --------------------------------------------------------------------------------------------------
def test_build_str_id_combines_statistic_and_qoi_name_otherwise() -> None:
    output_under_test = build(AcceptanceQoI(), RunningMeanStatistic())

    assert output_under_test.str_id == "mean of acceptance"


# --------------------------------------------------------------------------------------------------
def test_build_pads_str_id_to_minimum_width_twelve() -> None:
    output_under_test = build(MeanQoI(), IdentityStatistic())

    assert output_under_test.str_id == f"{'mean':<12}"
    assert len(output_under_test.str_id) == 12


# --------------------------------------------------------------------------------------------------
def test_build_widens_str_format_for_long_names() -> None:
    output_under_test = build(AcceptanceQoI(), RunningMeanStatistic())

    assert output_under_test.str_format == "<+18.3e"


# --------------------------------------------------------------------------------------------------
def test_build_output_is_always_logged() -> None:
    output_under_test = build(AcceptanceQoI(), IdentityStatistic())

    assert output_under_test.log
