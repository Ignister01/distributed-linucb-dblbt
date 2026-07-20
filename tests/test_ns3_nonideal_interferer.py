"""Structural regression for the non-ideal ns-3 interferer."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "ns3" / "scenarios" / "dblbt-nru-wifi-validation.cc"


def test_nonideal_interferer_registers_a_phased_array_with_3gpp_channel() -> None:
    text = SCENARIO.read_text(encoding="ascii")

    assert (
        "InstallInterferer (Ptr<MultiModelSpectrumChannel> channel,\n"
        "                   Ptr<ThreeGppSpectrumPropagationLossModel> "
        "spectrumPropagation,"
    ) in text
    assert "CreateObjectWithAttributes<UniformPlanarArray>" in text
    assert "spectrumPropagation->AddDevice (devices.Get (0), interferenceAntenna);" in text
    assert "InstallInterferer (channel, spectrumPropagation, appStart, simTime," in text
