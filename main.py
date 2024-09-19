import os
import json
import requests
import pandas as pd

from tqdm import tqdm
from datetime import datetime
from pydantic import BaseModel, Field, field_validator

from openai import OpenAI, pydantic_function_tool
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from bs4 import BeautifulSoup

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
HEADING_3 = 'mw-heading3'
HEADING_2 = 'mw-heading2'
SYSTEM_PROMPT = '''You are a data extraction assistant tasked with extracting information about hurricanes in 1975 from the provided text. You will be given unstructured text from a Wikipedia article about hurricanes and should convert it into the given structure. If the text mentions less significant systems or tropical depressions without specific headings, include those entries as well, but only extract data if it is available.'''


### Hurricanes Data

class HurricaneData(BaseModel):
    hurricane_storm_name: str
    date_start: str = Field(description='The start date of the hurricane in YYYY-MM-DD format')
    date_end: str = Field(description='The end date of the hurricane in YYYY-MM-DD format')
    number_of_deaths: int = Field(description='The total number of deaths caused by the hurricane')
    list_of_areas_affected: list[str] = Field(description='List of areas affected by the hurricane')

    @field_validator('date_start', 'date_end')
    def validate_date(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
        except ValueError:
            raise ValueError('Dates must be in YYYY-MM-DD format')
        return v

    @field_validator('number_of_deaths')
    def validate_deaths(cls, v):
        if v is not None and v < 0:
            raise ValueError('Number of deaths cannot be negative')
        return v
    
    @field_validator('list_of_areas_affected')
    def validate_areas(cls, value):
        if not isinstance(value, list) or not all(isinstance(area, str) for area in value):
            raise ValueError('list_of_areas_affected must be a list of strings')
        return value
    
class Hurricanes(BaseModel):
    hurricanes: list[HurricaneData]


### Scraping

def scrape_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup

def scrape_hurricanes():
    url = 'https://en.wikipedia.org/wiki/1975_Pacific_hurricane_season'
    soup = scrape_page(url)
    return soup
    
def extract_hurricane_section(hurricane_heading):
    content = [str(hurricane_heading)]
    next_node = hurricane_heading.find_next_sibling()
    
    while next_node:
        next_node_class = next_node.get('class', [])
        if HEADING_3 in next_node_class or HEADING_2 in next_node_class:
            break
        
        content.append(str(next_node))
        next_node = next_node.find_next_sibling()

    section_content = ''.join(content)
    
    return section_content

def html_to_plain_text(html):
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text(separator=' ', strip=True)


### Generation

def generate_gpt3_tool_call(client, schema, messages: list[ChatCompletionMessageParam]):
    completion = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=messages,
        max_completion_tokens=384,
        n=1,
        tools=[schema],
        tool_choice={
            'type': 'function',
            'function': {'name': schema['function']['name']},
        },
    )

    tool_call = completion.choices[0].message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)
    return arguments


### Validation

def validate_row(row):
    errors = []

    try:
        start_date = datetime.strptime(row['date_start'], '%Y-%m-%d')
        end_date = datetime.strptime(row['date_end'], '%Y-%m-%d')
        if start_date > end_date:
            errors.append('Start date is after end date')
    except (ValueError, TypeError):
        errors.append('Invalid date format')

    try:
        int(row['number_of_deaths'])
    except TypeError:
        errors.append('number_of_deaths is not an integer')


    try:
        all(isinstance(area, str) for area in row['list_of_areas_affected'])
    except TypeError:
        errors.append('Invalid list_of_areas_affected')


    errors_str = ', '.join(errors) if errors else None

    print(f"Row {row.tolist()} containes the following errors: {errors_str}") if errors_str else None

    return errors_str
    

def main():
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    
    hurricanes_tool_schema = pydantic_function_tool(Hurricanes)
    del hurricanes_tool_schema['function']['strict']
    
    soup = scrape_hurricanes()
    hurricane_headings = soup.find_all('div', class_=HEADING_3)
    
    
    hurricanes_data = []
    for hurricane_heading in tqdm(hurricane_headings):
        hurricane_section = html_to_plain_text(extract_hurricane_section(hurricane_heading))

        gpt_messages: list[ChatCompletionMessageParam] = [
            { 'role': 'system', 'content': SYSTEM_PROMPT },
            { 'role': 'user', 'content': f'Extract the hurricane data from the following text:\n\n{hurricane_section}' }
        ]

        response = generate_gpt3_tool_call(openai_client, hurricanes_tool_schema, gpt_messages)
        hurricane_data = response['hurricanes']
        hurricanes_data += hurricane_data
        
    hurricanes_data_df = pd.DataFrame(hurricanes_data)
    hurricanes_data_df['validation_errors'] = hurricanes_data_df.apply(validate_row, axis=1)
    hurricanes_data_df = hurricanes_data_df[hurricanes_data_df['validation_errors'].isna()]
    del hurricanes_data_df['validation_errors']
    hurricanes_data_df.to_csv('hurricanes_1975.csv', header=True, index=False)
        
if __name__ == '__main__':
    main()