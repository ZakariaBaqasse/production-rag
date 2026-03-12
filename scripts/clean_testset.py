"""Clean and process a test dataset by mapping adversarial questions to reference contexts, categorizing queries into table extraction types, and exporting the processed data.

This module:
- Loads JSON document pages and CSV test data
- Maps adversarial questions to specific page numbers from Llama Parse JSON
- Categorizes queries as single_cell_lookup, multi_row_reasoning, or adversarial_multi_hop
- Exports the processed testset with updated reference contexts and categories
"""

import pandas as pd
import json
import ast
import csv

# ==========================================
# 1. FILE PATHS
# ==========================================
# Use the file names of the documents you uploaded
INPUT_CSV = "./artifacts/testsets/curated_testset.csv"
INPUT_JSON = "./.cache/parsed/6fdff42cce00775643335e0ccb1dc1024070bb86208a2c734e9c09675ca3894a.json"
OUTPUT_CSV = "./artifacts/testsets/final_ragas_testset.csv"

# ==========================================
# 2. DEFINITIONS & MAPPINGS
# ==========================================

# Map adversarial questions to the exact page numbers in the Llama Parse JSON
adversarial_page_map = {
    "For an ESP32 dual-core chip operating at 160 MHz in Modem-sleep mode, how does the typical power consumption range compare to a single-core chip at the same frequency?": [
        30
    ],
    "What specific voltage and result conditions lead to a degradation in ADC accuracy when using an attenuation of 3?": [
        44,
        45,
    ],
    "If VDD_SDIO is connected to the same PCB net as VDD3P3_RTC, how is the internal LDO affected and what is the status of the internal 6 $\\Omega$ resistor?": [
        18,
        52,
    ],
    "How does the typical output impedance of the Wi-Fi radio differ between the QFN 6x6 and QFN 5x5 package sizes?": [
        54,
        55,
    ],
    "Which ESP32 models are restricted to an 85 °C operating temperature, and what is the technical reason for this limitation compared to other models?": [
        52
    ],
    "Explain how to force a 1.8 V VDD_SDIO voltage using eFuses, even if the MTDI strapping pin is pulled low at reset.": [
        24
    ],
}

single_cell_queries = [
    "What are the absolute maximum voltage and storage temperature ratings for the ESP32 according to the datasheet?",
    "What is the high-level input voltage threshold (VIH) and the high-level source current (IOH) for VDD3P3_CPU pins in the ESP32 DC characteristics?",
    "What is the Bluetooth Basic Data Rate receiver sensitivity and adjacent channel selectivity at F0+1 MHz for the ESP32?",
    "What is the BLE receiver sensitivity and co-channel C/I for the ESP32?",
    "What is the Bluetooth LE transmitter power control range and adjacent channel transmit power at F0±2 MHz on the ESP32?",
    "What are the ADC DNL and INL specifications, and what is the maximum sampling rate with the DIG controller?",
    "What is the ADC calibration error range when attenuation is set to 3 (atten=3) on the ESP32?",
    "Which cryptographic hardware acceleration features are included in the security specifications, particularly concerning SHA-2?",
    "As part of the hardware validation process for the new board design, could you please identify the core type, package dimensions, and permissible VDD_SDIO voltage settings specifically for the ESP32-D0WD-V3 as detailed in the comparison data?",
    "start addres and size of Internal SRAM 2",
    "What are the ESD protection ratings (HBM and CDM) for the ESP32?",
    "Which internal memory segment is accessible by the ULP coprocessor during Deep-sleep mode?",
]

multi_row_queries = [
    "How much current does the ESP32 consume when transmitting 802.11b at 1 Mbps versus 802.11n MCS7?",
    "What is the Wi-Fi receiver sensitivity for 802.11b at 1 Mbps and for 802.11n HT20 MCS7 on the ESP32?",
    "What is the output impedance of the ESP32 Wi-Fi radio, and how does it differ between QFN package sizes?",
    "What is the minimum recommended power supply voltage for ESP32 chips, and why are there two different values listed?",
    "what chips is listed inside the ESP32 Series datasheet 5.2 and which ones is NRND?",
    "ESP32-D0WD-V3 maximum CPU frequency specification and power consumption in modem-sleep mode for dual-core chips running at 240 MHz",
    "ESP32 functional block diagram components and technology specs",
    "Can you detail the technical specifications regarding Bluetooth LE compliance, sensitivity, and controller capabilities listed in the features?",
    "if i use VDD_SDIO 1.8 V for external flash power supply, what resistor i should add and do i look at ESP32 Hardware Design Guidelines for circuit?",
    "what voltage ESP32-D0WDRH2-V3 need and how connect VDD_SDIO?",
]

# ==========================================
# 3. DATA PROCESSING
# ==========================================


def safe_parse_context(context_str):
    """Ensures reference_contexts are properly formatted JSON arrays of strings."""
    if pd.isna(context_str):
        return json.dumps([])
    context_str = str(context_str).strip()
    try:
        if context_str.startswith("[") and context_str.endswith("]"):
            parsed = ast.literal_eval(context_str)
            return json.dumps(parsed)
        return json.dumps([context_str])
    except Exception:
        return json.dumps([context_str])


def process_testset():
    """Process and clean the test dataset by mapping questions to contexts and categorizing queries.

    Loads a JSON file containing parsed document pages and a CSV test dataset. For each test case,
    it maps adversarial questions to their corresponding page contexts from the JSON, categorizes
    table extraction queries as single_cell_lookup or multi_row_reasoning, and ensures proper
    JSON formatting of reference contexts. Exports the processed dataset to a new CSV file.

    Raises:
        FileNotFoundError: If INPUT_JSON or INPUT_CSV files are not found.
        json.JSONDecodeError: If INPUT_JSON is not valid JSON.
    """
    # 1. Load JSON and map page numbers to their full markdown content
    with open(INPUT_JSON, encoding="utf-8") as f:
        parsed_pages = json.load(f)
    page_dict = {page["page_number"]: page["markdown"] for page in parsed_pages}

    # 2. Load CSV
    df = pd.read_csv(INPUT_CSV, sep=",", engine="python", on_bad_lines="skip")

    # Fix injected headers if they still exist
    df = df[df["user_input"] != "user_input"].copy()
    if "adversarial" in df.columns:
        df["question_category"] = df["question_category"].fillna(df["adversarial"])
        df = df.drop(columns=["adversarial"])

    df["reference_contexts"] = df["reference_contexts"].apply(safe_parse_context)

    # 3. Apply updates iteratively
    for idx, row in df.iterrows():
        # Strip string to avoid trailing comma/space mismatches
        query = str(row["user_input"]).strip().rstrip(",")

        # A. Inject realistic adversarial contexts directly from Llama Parse JSON
        if query in adversarial_page_map:
            pages = adversarial_page_map[query]
            # Extract full markdown text for each required page to simulate real chunks
            realistic_context = [page_dict[p] for p in pages]
            df.at[idx, "reference_contexts"] = json.dumps(realistic_context)
            df.at[idx, "question_category"] = "adversarial_multi_hop"

        # B. Split all table extractions into the new categories
        if query in single_cell_queries:
            df.at[idx, "question_category"] = "table_single_cell_lookup"
        elif query in multi_row_queries:
            df.at[idx, "question_category"] = "table_multi_row_reasoning"

    # Check to verify no 'table_extraction' values remain
    remaining = df[df["question_category"] == "table_extraction"]
    if not remaining.empty:
        print(f"Warning: {len(remaining)} items are still marked as 'table_extraction'")

    # 4. Export
    df.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_ALL)
    print(f"✅ Successfully cleaned testset and saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    process_testset()
