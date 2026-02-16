#!/usr/bin/env python3
"""Generate the curated RAGAS testset CSV.

Writes 18 manually curated questions grounded in actual ESP32 datasheet content
to artifacts/ragas/curated_testset.csv.  The CSV follows the same schema as
baseline_testset.csv produced by the RAGAS testset generator.
"""

import csv
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
PARSED = (
    ROOT
    / ".cache/parsed/6fdff42cce00775643335e0ccb1dc1024070bb86208a2c734e9c09675ca3894a.json"
)
OUT = ROOT / "artifacts/ragas/curated_testset.csv"

with open(PARSED) as f:
    pages = json.load(f)


def ctx(page_nums: list[int]) -> str:
    """Build a reference_contexts string (Python list repr) from page numbers."""
    parts = [pages[p]["markdown"] for p in page_nums]
    return repr(parts)  # e.g. "['markdown for p1', 'markdown for p2']"


questions = [
    # ── Category 1: Hard table extraction (electrical / radio specs) ────────
    {
        "user_input": "What are the absolute maximum voltage and storage temperature ratings for the ESP32 according to the datasheet?",
        "reference_contexts": ctx([51]),
        "reference": (
            "According to Table 5-1, the absolute maximum allowed input voltage for "
            "VDDA, VDD3P3, VDD3P3_RTC, VDD3P3_CPU, and VDD_SDIO is -0.3 V min to 3.6 V max. "
            "The cumulative IO output current (I_output) maximum is 1200 mA. "
            "The storage temperature (T_STORE) range is -40 °C min to 150 °C max."
        ),
        "persona_name": "Hardware Design Engineer",
        "query_style": "MISSPELLED",
        "query_length": "MEDIUM",
    },
    {
        "user_input": "What is the high-level input voltage threshold (VIH) and the high-level source current (IOH) for VDD3P3_CPU pins in the ESP32 DC characteristics?",
        "reference_contexts": ctx([52]),
        "reference": (
            "From Table 5-3, VIH (high-level input voltage) has a minimum of 0.75 × VDD "
            "and a maximum of VDD + 0.3 V. The high-level source current IOH for the "
            "VDD3P3_CPU power domain is typically 40 mA (at VDD = 3.3 V, VOH >= 2.64 V, "
            "output drive strength set to maximum). Per footnote 2, this per-pin current "
            "is gradually reduced from ~40 mA to ~29 mA as the number of current-source pins increases."
        ),
        "persona_name": "Hardware Design Engineer",
        "query_style": "PERFECT_GRAMMAR",
        "query_length": "LONG",
    },
    {
        "user_input": "How much current does the ESP32 consume when transmitting 802.11b at 1 Mbps versus 802.11n MCS7?",
        "reference_contexts": ctx([52]),
        "reference": (
            "According to Table 5-4, transmitting 802.11b DSSS at 1 Mbps with POUT = +19.5 dBm "
            "draws a typical current of 240 mA. Transmitting 802.11n OFDM MCS7 with "
            "POUT = +14 dBm draws a typical current of 180 mA. So 11b transmission "
            "consumes 60 mA more than 11n MCS7."
        ),
        "persona_name": "IoT Power Engineer",
        "query_style": "PERFECT_GRAMMAR",
        "query_length": "MEDIUM",
    },
    {
        "user_input": "What is the Wi-Fi receiver sensitivity for 802.11b at 1 Mbps and for 802.11n HT20 MCS7 on the ESP32?",
        "reference_contexts": ctx([53, 54]),
        "reference": (
            "From Table 5-6, the typical sensitivity for 802.11b at 1 Mbps is -98 dBm, "
            "and the typical sensitivity for 802.11n HT20 MCS7 is -73 dBm. "
            "The difference is 25 dB."
        ),
        "persona_name": "RF Systems Engineer",
        "query_style": "PERFECT_GRAMMAR",
        "query_length": "MEDIUM",
    },
    {
        "user_input": "What is the output impedance of the ESP32 Wi-Fi radio, and how does it differ between QFN package sizes?",
        "reference_contexts": ctx([54]),
        "reference": (
            "According to footnote 2 under Table 5-6, the typical Wi-Fi radio output impedance "
            "differs by package: for chips in a QFN 6×6 package the value is 30+j10 Ω, "
            "and for chips in a QFN 5×5 package the value is 35+j10 Ω."
        ),
        "persona_name": "RF Systems Engineer",
        "query_style": "PERFECT_GRAMMAR",
        "query_length": "MEDIUM",
    },
    {
        "user_input": "What is the Bluetooth Basic Data Rate receiver sensitivity and adjacent channel selectivity at F0+1 MHz for the ESP32?",
        "reference_contexts": ctx([54]),
        "reference": (
            "From Table 5-7, the BT Basic Data Rate receiver sensitivity at 0.1% BER "
            "is -90 dBm min / -89 dBm typ / -88 dBm max. "
            "The adjacent channel selectivity C/I at F = F0 + 1 MHz is <= -6 dB (max)."
        ),
        "persona_name": "RF Systems Engineer",
        "query_style": "PERFECT_GRAMMAR",
        "query_length": "LONG",
    },
    {
        "user_input": "What is the BLE receiver sensitivity and co-channel C/I for the ESP32?",
        "reference_contexts": ctx([56]),
        "reference": (
            "From Table 5-11, the BLE receiver sensitivity at 30.8% PER is "
            "-94 dBm min / -93 dBm typ / -92 dBm max. "
            "The co-channel C/I is typically +10 dB."
        ),
        "persona_name": "Embedded Firmware Engineer",
        "query_style": "PERFECT_GRAMMAR",
        "query_length": "SHORT",
    },
    {
        "user_input": "What is the Bluetooth LE transmitter power control range and adjacent channel transmit power at F0±2 MHz on the ESP32?",
        "reference_contexts": ctx([58]),
        "reference": (
            "From Table 5-12, the BLE RF power control range is -12 dBm min to +9 dBm max. "
            "The adjacent channel transmit power at F = F0 ± 2 MHz is typically -52 dBm. "
            "At F = F0 ± 3 MHz it is typically -58 dBm, and at F0 ± > 3 MHz it is typically -60 dBm."
        ),
        "persona_name": "RF Systems Engineer",
        "query_style": "PERFECT_GRAMMAR",
        "query_length": "LONG",
    },
    {
        "user_input": "What are the ADC DNL and INL specifications, and what is the maximum sampling rate with the DIG controller?",
        "reference_contexts": ctx([43]),
        "reference": (
            "From Table 4-3, the ADC DNL (differential nonlinearity) is -7 to +7 LSB, "
            "and the INL (integral nonlinearity) is -12 to +12 LSB. These are measured "
            "with the RTC controller, ADC connected to an external 100 nF capacitor, "
            "DC signal input, at 25 °C with Wi-Fi & Bluetooth off. "
            "The maximum sampling rate with the DIG controller is 2 Msps; "
            "with the RTC controller it is 200 ksps."
        ),
        "persona_name": "Analog Design Engineer",
        "query_style": "PERFECT_GRAMMAR",
        "query_length": "MEDIUM",
    },
    {
        "user_input": "What is the ADC calibration error range when attenuation is set to 3 (atten=3) on the ESP32?",
        "reference_contexts": ctx([44]),
        "reference": (
            "From Table 4-4, at Atten = 3 (effective measurement range 150 ~ 2450 mV), "
            "the total error after calibration using eFuse Vref is -60 mV to +60 mV. "
            "Additionally, the note states that when atten = 3 and the measurement result "
            "is above 3000 (voltage at approx. 2450 mV), the ADC accuracy will be worse."
        ),
        "persona_name": "Analog Design Engineer",
        "query_style": "PERFECT_GRAMMAR",
        "query_length": "MEDIUM",
    },
    # ── Category 2: Pin mapping / peripheral config ─────────────────────────
    {
        "user_input": "Which GPIO pins are mapped to touch sensor signals T5 and T8 on the ESP32?",
        "reference_contexts": ctx([45]),
        "reference": (
            "According to Table 4-5, touch sensor signal T5 is mapped to pin MTDI, "
            "and T8 is mapped to pin 32K_XN."
        ),
        "persona_name": "Embedded Firmware Engineer",
        "query_style": "PERFECT_GRAMMAR",
        "query_length": "SHORT",
    },
    {
        "user_input": "Which pins are ADC2_CH5 and ADC1_CH4 connected to on the ESP32, and what other peripheral functions share those pins?",
        "reference_contexts": ctx([46]),
        "reference": (
            "From Table 4-6, ADC2_CH5 is connected to pin MTDI, and ADC1_CH4 is connected "
            "to pin 32K_XP. Looking at the same table, MTDI also serves as TOUCH5 "
            "(capacitive touch sensor) and as the MTDI JTAG signal. Pin 32K_XP also "
            "serves as TOUCH9."
        ),
        "persona_name": "Hardware Design Engineer",
        "query_style": "PERFECT_GRAMMAR",
        "query_length": "LONG",
    },
    {
        "user_input": "In the ESP32 GPIO_Matrix, what is signal number 63 and does it have a corresponding IO_MUX core input?",
        "reference_contexts": ctx([64]),
        "reference": (
            "In Table GPIO_Matrix, signal number 63 has input signal VSPICLK_in "
            "(default value 0 if unassigned), output signal VSPICLK_out_mux, "
            "and output enable VSPICLK_oe. The 'Same Input Signal from IO_MUX Core' "
            "column is 'yes', meaning this signal has a direct IO_MUX path."
        ),
        "persona_name": "Embedded Firmware Engineer",
        "query_style": "PERFECT_GRAMMAR",
        "query_length": "MEDIUM",
    },
    {
        "user_input": "What are the ESD protection ratings (HBM and CDM) for the ESP32?",
        "reference_contexts": ctx([53]),
        "reference": (
            "From Table 5-5 (Reliability Qualifications), the ESP32 ESD ratings are: "
            "HBM (Human Body Mode) ± 2000 V (per JS-001 standard), and "
            "CDM (Charge Device Mode) ± 500 V (per JS-002 standard). "
            "Footnote 1 notes that JEDEC JEP155 states 500 V HBM allows safe manufacturing "
            "with a standard ESD control process, and footnote 2 notes that JEP157 "
            "states 250 V CDM allows safe manufacturing."
        ),
        "persona_name": "Hardware Design Engineer",
        "query_style": "PERFECT_GRAMMAR",
        "query_length": "SHORT",
    },
    # ── Category 3: Cross-section synthesis (require 2+ sections) ───────────
    {
        "user_input": "Which ESP32 GPIO pins support both ADC and capacitive touch sensing simultaneously?",
        "reference_contexts": ctx([45, 46]),
        "reference": (
            "By cross-referencing Table 4-5 (touch sensor GPIOs) and Table 4-6 "
            "(peripheral pin configurations), the pins that support both ADC and "
            "touch sensing are: GPIO4 (ADC2_CH0 / TOUCH0), GPIO0 (ADC2_CH1 / TOUCH1), "
            "GPIO2 (ADC2_CH2 / TOUCH2), MTDO (ADC2_CH3 / TOUCH3), "
            "MTCK (ADC2_CH4 / TOUCH4), MTDI (ADC2_CH5 / TOUCH5), "
            "MTMS (ADC2_CH6 / TOUCH6), GPIO27 (ADC2_CH7 / TOUCH7), "
            "32K_XN (ADC1_CH5 / TOUCH8), and 32K_XP (ADC1_CH4 / TOUCH9). "
            "All 10 touch-capable pins also have ADC functionality."
        ),
        "persona_name": "Embedded Firmware Engineer",
        "query_style": "PERFECT_GRAMMAR",
        "query_length": "MEDIUM",
    },
    {
        "user_input": "The MTDI pin on the ESP32 is a strapping pin. What voltage does it control, and what ADC and touch sensor functions does it also serve?",
        "reference_contexts": ctx([21, 45, 46]),
        "reference": (
            "MTDI is a strapping pin that controls the VDD_SDIO voltage at reset: "
            "when MTDI = 0 (default, pulled down), VDD_SDIO is powered from VDD3P3_RTC "
            "(typically 3.3 V); when MTDI = 1, VDD_SDIO is powered from internal 1.8 V LDO. "
            "From Table 4-6, MTDI is also mapped to ADC2_CH5, and from Table 4-5, "
            "MTDI is mapped to touch sensor signal T5. It also serves as the MTDI JTAG signal."
        ),
        "persona_name": "Hardware Design Engineer",
        "query_style": "PERFECT_GRAMMAR",
        "query_length": "LONG",
    },
    {
        "user_input": "How does the ESP32 BLE receiver sensitivity compare to the Bluetooth Classic Basic Data Rate sensitivity?",
        "reference_contexts": ctx([54, 56]),
        "reference": (
            "The BLE receiver sensitivity (Table 5-11) at 30.8% PER is "
            "-94 dBm min / -93 dBm typ / -92 dBm max. "
            "The BT Basic Data Rate receiver sensitivity (Table 5-7) at 0.1% BER is "
            "-90 dBm min / -89 dBm typ / -88 dBm max. "
            "Thus the BLE receiver is about 4 dB more sensitive than BT Classic Basic Data Rate, "
            "though at a different error rate criterion (30.8% PER vs 0.1% BER)."
        ),
        "persona_name": "RF Systems Engineer",
        "query_style": "PERFECT_GRAMMAR",
        "query_length": "MEDIUM",
    },
    {
        "user_input": "What is the minimum recommended power supply voltage for ESP32 chips, and why are there two different values listed?",
        "reference_contexts": ctx([51]),
        "reference": (
            "From Table 5-2, the minimum recommended voltage for VDDA, VDD3P3_RTC, "
            "VDD3P3, and VDD_SDIO (3.3 V mode) shows two values: 2.3 V and 3.0 V. "
            "Footnote 2 explains the difference: chips with a 3.3 V flash or PSRAM "
            "in-package have a minimum of 3.0 V, while chips with no flash or PSRAM "
            "in-package have a minimum of 2.3 V. VDD3P3_CPU has a separate minimum "
            "of 1.8 V with a typical of 3.3 V and max of 3.6 V."
        ),
        "persona_name": "Power Supply Engineer",
        "query_style": "PERFECT_GRAMMAR",
        "query_length": "MEDIUM",
    },
]

OUT.parent.mkdir(parents=True, exist_ok=True)
FIELDS = [
    "user_input",
    "reference_contexts",
    "reference",
    "persona_name",
    "query_style",
    "query_length",
    "synthesizer_name",
]

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS, quoting=csv.QUOTE_ALL)
    w.writeheader()
    for q in questions:
        q.setdefault("synthesizer_name", "manual_curation")
        w.writerow(q)

print(f"Wrote {len(questions)} curated questions → {OUT}")
