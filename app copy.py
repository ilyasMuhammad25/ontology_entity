from flask import Flask, jsonify,request,Response,render_template, request
from rdflib import Graph, URIRef, Literal, RDF
import networkx as nx
import requests
from sqlalchemy import create_engine, inspect,text
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import json
from openai import OpenAI

app = Flask(__name__)

# Load database configuration from a file
def load_db_config():
    try:
        with open('db_config.json', 'r') as config_file:
            return json.load(config_file)
    except FileNotFoundError:
        return {
            'host': 'localhost',
            'username': 'root',
            'password': '',
            'dbname': 'bps'
        }

db_config = load_db_config()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/config')
def config():
    return render_template('config.html')

@app.route('/save_config', methods=['POST'])
def save_config():
    global db_config
    data = request.get_json()

    db_config['host'] = data['host']
    db_config['username'] = data['username']
    db_config['password'] = data['password']
    db_config['dbname'] = data['dbname']

    # Save the configuration to a file
    with open('db_config.json', 'w') as config_file:
        json.dump(db_config, config_file)

    return jsonify({'message': 'Configuration saved successfully'})

# Function to get metadata from the database
# Function to get metadata from the database
# Function to get database metadata
def get_database_metadata():
    # Load the database configuration
    db_config = load_db_config()

    # Create connection string dynamically based on config
    connection_string = f"mysql+pymysql://{db_config['username']}:{db_config['password']}@{db_config['host']}/{db_config['dbname']}"
    
    # Create a database engine
    engine = create_engine(connection_string)
    
    # Use SQLAlchemy's inspector to get metadata
    inspector = inspect(engine)

    metadata = {}

    # Iterate through all tables in the database
    for table_name in inspector.get_table_names():
        columns = inspector.get_columns(table_name)
        relationships = {}  # Placeholder for relationships if needed

        # Store table metadata
        metadata[table_name] = {
            'fields': [column['name'] for column in columns],
            'relations': relationships  # Modify this as needed for actual relationships
        }

    return metadata

# Function to identify relationships between entities
def identify_relationships(metadata):
    relationships = []
    for table, data in metadata.items():
        for field, relation in data.get('relations', {}).items():
            relationships.append((table, relation.split('.')[0], field))
    return relationships

# Build ontology using NetworkX
def build_local_ontology(metadata, relationships):
    G = nx.DiGraph()
    
    # Add nodes and attributes from metadata
    for table, data in metadata.items():
        G.add_node(table, type='entity')
        for field in data['fields']:
            G.add_node(field, type='attribute')
            G.add_edge(table, field, type='has_attribute')

    # Add relationships between entities
    for (src, dst, field) in relationships:
        G.add_edge(src, dst, type=f'relationship_{field}')
    
    return G

# Convert ontology to RDF
def convert_to_rdf(ontology):
    g = Graph()

    for node, data in ontology.nodes(data=True):
        uri = URIRef(f"http://example.org/{node}")
        g.add((uri, RDF.type, Literal(data.get('type'))))

    for edge in ontology.edges(data=True):
        subject = URIRef(f"http://example.org/{edge[0]}")
        obj = URIRef(f"http://example.org/{edge[1]}")
        g.add((subject, URIRef(f"http://example.org/{edge[2]['type']}"), obj))
    
    return g

# Save RDF graph to file
def save_rdf_to_file(filename, rdf_graph):
    rdf_graph.serialize(destination=filename, format='turtle')



@app.route('/construct_ontology', methods=['POST'])
def construct_ontology():
    metadata = get_database_metadata()
    relationships = identify_relationships(metadata)
    ontology = build_local_ontology(metadata, relationships)
    rdf_graph = convert_to_rdf(ontology)

    # Save RDF graph to file
    save_rdf_to_file('ontology2.ttl', rdf_graph)

    # Convert ontology to vis.js format
    nodes = [{'id': node, 'label': node, 'group': 'entity' if data.get('type') == 'entity' else 'attribute'} 
             for node, data in ontology.nodes(data=True)]
    edges = [{'from': edge[0], 'to': edge[1], 'label': ontology.edges[edge].get('type', '')} 
             for edge in ontology.edges()]

    return jsonify({'nodes': nodes, 'edges': edges})


# Function to handle model generation request
def generate(payload):
    try:
        # Send the POST request to the model API
        with requests.post('http://localhost:11434/api/generate', json=payload, stream=True) as response:
            response.raise_for_status()
            full_response = ""
            # Stream the response line by line
            for line in response.iter_lines():
                if line:
                    json_response = json.loads(line)
                    if 'response' in json_response:
                        full_response += json_response['response']
                        yield json_response['response']
            
            # Yield the full response
            yield f"\n\nFull response: {full_response}"
    
    except requests.RequestException as e:
        yield f"Error: {str(e)}"

# Route to create prompt and send to model API
@app.route('/ask', methods=['GET'])
def ask_ollama():
    # Get the database metadata
    metadata = get_database_metadata()
    
    # Build the prompt dynamically
    prompt = "Saya punya tabel-tabel berikut:\n\n"
    
    for table, info in metadata.items():
        prompt += f"CREATE TABLE {table} (\n"
        prompt += ",\n".join([f"  {field} varchar(255)" for field in info['fields']])
        prompt += "\n);\n\n"

    prompt += "Buatkan ontologi .ttl-nya kurang lebih seperti ini."

    # Prepare the payload for the AI model
    payload = {
        "model": "llama3.2",
        "prompt": prompt,
        "stream": True
    }

    # Use the generate function to stream the response
    return Response(generate(payload), content_type='text/plain')
    
   

# cek entity resolution
@app.route('/entity')
def entity():
    return render_template('entity.html')
@app.route('/get_tables', methods=['GET'])
def get_tables():
    db_config = load_db_config()
    connection_string = f"mysql+pymysql://{db_config['username']}:{db_config['password']}@{db_config['host']}/{db_config['dbname']}"
    engine = create_engine(connection_string)

    inspector = inspect(engine)
    tables = inspector.get_table_names()
    return jsonify({'tables': tables})

@app.route('/get_columns/<table_name>', methods=['GET'])
def get_columns(table_name):
    db_config = load_db_config()
    connection_string = f"mysql+pymysql://{db_config['username']}:{db_config['password']}@{db_config['host']}/{db_config['dbname']}"
    engine = create_engine(connection_string)

    inspector = inspect(engine)
    columns = [column['name'] for column in inspector.get_columns(table_name)]
    return jsonify({'columns': columns})


@app.route('/check_entity_resolution', methods=['POST'])
def check_entity_resolution():
    table1 = request.form.get('table1')
    column1 = request.form.get('columns1')
    table2 = request.form.get('table2')
    column2 = request.form.get('columns2')

    # Load the database configuration
    db_config = load_db_config()
    connection_string = f"mysql+pymysql://{db_config['username']}:{db_config['password']}@{db_config['host']}/{db_config['dbname']}"
    
    # Create a database engine
    engine = create_engine(connection_string)

    # Fetch data from the first table and column
    with engine.connect() as connection:
        # Query untuk mengambil nilai dari kolom pertama
        query1 = text(f"SELECT DISTINCT `{column1}` FROM `{table1}`")
        result1 = connection.execute(query1)
        values1 = [row[0] for row in result1]

        # Query untuk mengambil nilai dari kolom kedua
        query2 = text(f"SELECT DISTINCT `{column2}` FROM `{table2}`")
        result2 = connection.execute(query2)
        values2 = [row[0] for row in result2]

    # Create prompt for Ollama with similarity scoring
    prompt = "For the following pairs of entities, provide a similarity score (0-1) and whether they refer to the same item:\n"
    for value1 in values1:
        for value2 in values2:
            prompt += f"- '{value1}' and '{value2}'\n"

    # Call Ollama API
    response = call_ollama_api(prompt)
    
    # Format the response
    formatted_response = format_response(response)
    
    return jsonify({'result': formatted_response})

def call_ollama_api(prompt):
    payload = {
        "model": "llama3.2",
        "prompt": prompt,
        "stream": True
    }

    try:
        with requests.post('http://localhost:11434/api/generate', json=payload, stream=True) as response:
            response.raise_for_status()
            full_response = ""
            for line in response.iter_lines():
                if line:
                    json_response = json.loads(line)
                    if 'response' in json_response:
                        full_response += json_response['response']
            return full_response
        
    except requests.RequestException as e:
        return f"Error: {str(e)}"

def format_response(response):
    # Assuming the response is in the format you expect
    formatted = []
    lines = response.splitlines()
    for line in lines:
        # Example line processing; adapt as necessary
        if line.strip():  # Ignore empty lines
            formatted.append(line.strip())
    return formatted

@app.route('/check_entity_resolution2', methods=['POST'])
def check_entity_resolution2():
    table1 = request.form.get('table1')
    column1 = request.form.get('columns1')
    table2 = request.form.get('table2')
    column2 = request.form.get('columns2')

    # Load the database configuration
    db_config = load_db_config()
    connection_string = f"mysql+pymysql://{db_config['username']}:{db_config['password']}@{db_config['host']}/{db_config['dbname']}"
    
    # Create a database engine
    engine = create_engine(connection_string)

    # Fetch data from the first table and column
    with engine.connect() as connection:
        # Query to fetch values from the first column
        query1 = text(f"SELECT DISTINCT `{column1}` FROM `{table1}`")
        result1 = connection.execute(query1)
        values1 = [row[0] for row in result1]

        # Query to fetch values from the second column
        query2 = text(f"SELECT DISTINCT `{column2}` FROM `{table2}`")
        result2 = connection.execute(query2)
        values2 = [row[0] for row in result2]

    # Perform fuzzy matching
    matches = []
    threshold = 80  # Set a threshold for matching score

    for value1 in values1:
        # Get the best match for each value1 from values2
        best_match, score = process.extractOne(value1, values2, scorer=fuzz.token_sort_ratio)
        if score >= threshold:
            matches.append((value1, best_match, score))

    # Prepare the response
    response_message = (
        f"Matches found: {len(matches)}\n"
        f"Detailed Matches:\n"
    )
    
    for value1, best_match, score in matches:
        response_message += f"- '{value1}' matches with '{best_match}' (Score: {score})\n"

    return jsonify({'result': response_message})

if __name__ == '__main__':
    app.run(debug=True)
