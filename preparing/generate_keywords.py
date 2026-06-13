import csv
import openai
import re
from tqdm import tqdm
import os
import time

client = openai.OpenAI(api_key='YOUR_OpenAI_API_KEY')

def process_report_with_retry(report, max_retries=5):
    retries = 0
    while retries < max_retries:
        return process_report(report)

def process_report(report):
    prompt = f"""
# You are a pathology assistant. You are given an pathology report describing pathology WSIs.\n
# Ignore all information except for the part that can be detected based on only H&E stained WSIs\n
# Based on the given pathology report, extract the main keywords that are observable only in H&E stained patch images at 20x magnification.\n
# Provide a maximum of 10 distinct, non-overlapping, and contrasting keywords.\n
# Please separate each keyword with a comma in the output.\n

# Report: \n{report}\n
"""
    response = client.chat.completions.create(
        model="gpt-5-mini-2025-08-07",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=1,
        max_completion_tokens=4096,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content


def main():
    input_file = 'ORIGIN_REPORT_CSV_FILE_PATH'
    output_file = 'PATH_TO_SAVE_KEYWORD_CSV'

    # Load created file for re-start
    processed_reports = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', newline='', encoding='utf-8') as outfile:
            reader = csv.DictReader(outfile)
            for row in reader:
                processed_reports.add(row['p_id'])

    # For un-processed rows
    with open(input_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        reader = list(reader)

        # For TCGA
        required_headers = ['p_id','report']
        fieldnames = required_headers + ['keywords']

        with open(output_file, 'a', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

            for row in tqdm(reader):  # , total=len(list(reader))):
                # If already processed
                if row['p_id'] in processed_reports:
                    continue

                # Process the report and add the processed report field
                report = row['report']
                keywords = process_report_with_retry(report)

                # Create a new dictionary with only the required fields
                filtered_row = {field: row[field] for field in required_headers}
                filtered_row['keywords'] = keywords

                writer.writerow(filtered_row)
                outfile.flush()

if __name__ == "__main__":
    main()