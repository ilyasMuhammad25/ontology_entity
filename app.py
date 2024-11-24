from flask import Flask, jsonify,request,Response,render_template, request
from rdflib import Graph, Namespace, Literal, URIRef, BNode
from rdflib.namespace import RDF, RDFS, OWL, XSD
import networkx as nx
import requests
from difflib import SequenceMatcher
import re
import itertools
from sqlalchemy import create_engine, inspect,text
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import json
import os
import pandas as pd
from entity_ontology import OntologyBuilder
import google.generativeai as genai
import mysql.connector


app = Flask(__name__)
api_key = "AIzaSyA1ZHwqVwzj1z9A8lWNhlXZfX5-sby__8A"
ontology_builder = OntologyBuilder()
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

def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="newbps"
    )

def create_gsbpm_ontology():
    # Initialize RDF graph
    g = Graph()
    
    # Define namespaces
    GSBPM = Namespace("http://example.org/gsbpm#")
    g.bind("gsbpm", GSBPM)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("owl", OWL)
    g.bind("xsd", XSD)
    
    # Define Ontology
    ontology = URIRef("http://example.org/gsbpm#GSBPMOntology")
    g.add((ontology, RDF.type, OWL.Ontology))
    g.add((ontology, RDFS.label, Literal("GSBPM Ontology", lang="en")))
    g.add((ontology, RDFS.comment, Literal("An ontology for the Generic Statistical Business Process Model", lang="en")))
    
    # Define Main Classes
    classes = [
        ("StatisticalProject", "Statistical Project", "A statistical project following GSBPM model"),
        ("SpecifyNeeds", "Specify Needs", "Phase 1 of GSBPM - Specify Needs"),
        ("Design", "Design", "Phase 2 of GSBPM - Design"),
        ("Build", "Build", "Phase 3 of GSBPM - Build"),
        ("Collect", "Collect", "Phase 4 of GSBPM - Collect"),
        ("Process", "Process", "Phase 5 of GSBPM - Process"),
        ("Analyze", "Analyze", "Phase 6 of GSBPM - Analyze"),
        ("Disseminate", "Disseminate", "Phase 7 of GSBPM - Disseminate"),
        ("QualityManagement", "Quality Management", "Overarching process - Quality Management"),
        ("MetadataManagement", "Metadata Management", "Overarching process - Metadata Management")
    ]
    
    for class_id, label, comment in classes:
        class_uri = GSBPM[class_id]
        g.add((class_uri, RDF.type, OWL.Class))
        g.add((class_uri, RDFS.label, Literal(label, lang="en")))
        g.add((class_uri, RDFS.comment, Literal(comment, lang="en")))
    
    # Define Properties
    g.add((GSBPM.hasProjectID, RDF.type, OWL.DatatypeProperty))
    g.add((GSBPM.hasProjectID, RDFS.domain, GSBPM.StatisticalProject))
    g.add((GSBPM.hasProjectID, RDFS.range, XSD.string))
    
    g.add((GSBPM.hasProjectName, RDF.type, OWL.DatatypeProperty))
    g.add((GSBPM.hasProjectName, RDFS.domain, GSBPM.StatisticalProject))
    g.add((GSBPM.hasProjectName, RDFS.range, XSD.string))
    
    # Connect to database and add instances
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        # Fetch statistical activities and create instances
        cursor.execute("SELECT * FROM master_kegiatans")
        activities = cursor.fetchall()
        
        for activity in activities:
            # Create Statistical Project instance
            project_uri = URIRef(f"{GSBPM}Project_{activity['id']}")
            g.add((project_uri, RDF.type, GSBPM.StatisticalProject))
            g.add((project_uri, GSBPM.hasProjectID, Literal(str(activity['id']))))
            g.add((project_uri, GSBPM.hasProjectName, Literal(activity['judul_kegiatan'])))
            
            if activity['created_at']:
                g.add((project_uri, GSBPM.hasStartDate, 
                      Literal(activity['created_at'], datatype=XSD.dateTime)))
            
            # Add Design phase information
            design_node = BNode()
            g.add((project_uri, GSBPM.hasDesign, design_node))
            g.add((design_node, RDF.type, GSBPM.Design))
            g.add((design_node, GSBPM.statisticalType, Literal(activity['jenis_statistik'])))
            g.add((design_node, GSBPM.dataCollectionMethod, Literal(activity['cara_pengumpulan_data'])))
            
            # Fetch and add indicators as part of Specify Needs
            cursor.execute("SELECT * FROM master_indikators WHERE id_mskeg = %s", (activity['id'],))
            indicators = cursor.fetchall()
            
            if indicators:
                needs_node = BNode()
                g.add((project_uri, GSBPM.hasSpecifyNeeds, needs_node))
                g.add((needs_node, RDF.type, GSBPM.SpecifyNeeds))
                
                for indicator in indicators:
                    g.add((needs_node, GSBPM.identifyNeeds, Literal(indicator['nama'])))
                    g.add((needs_node, GSBPM.consultStakeholders, Literal(indicator['produsen_data_name'])))
            
            # Add metadata management information
            metadata_node = BNode()
            g.add((project_uri, GSBPM.hasMetadataManagement, metadata_node))
            g.add((metadata_node, RDF.type, GSBPM.MetadataManagement))
            
            # Fetch and add variables as part of metadata
            cursor.execute("SELECT * FROM metadata_variabels WHERE id_mskeg = %s", (activity['id'],))
            variables = cursor.fetchall()
            
            for variable in variables:
                g.add((metadata_node, GSBPM.hasVariable, Literal(variable['nama'])))
                g.add((metadata_node, GSBPM.hasDefinition, Literal(variable['definisi'])))
        
    finally:
        cursor.close()
        conn.close()
    
    return g

app = Flask(__name__)

@app.route('/gsbpm-ontology')
def get_gsbpm_ontology():
    """
    Generate and return GSBPM ontology in Turtle format
    """
    try:
        # Create the ontology graph
        g = create_gsbpm_ontology()
        
        # Serialize to Turtle format
        turtle_data = g.serialize(format='turtle')
        
        # Return as response with appropriate content type
        return Response(
            turtle_data,
            mimetype='text/turtle',
            headers={
                'Content-Disposition': 'attachment; filename=gsbpm_ontology.ttl'
            }
        )
    except Exception as e:
        return str(e), 500



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
    # Create graph
    g = Graph()
    
    # Define namespaces
    mfg = Namespace("http://example.org/manufacturing#")
    rr = Namespace("http://www.w3.org/ns/r2rml#")
    sim = Namespace("http://example.org/similarity#")
    
    # Bind prefixes
    g.bind('rdf', RDF)
    g.bind('rdfs', RDFS)
    g.bind('owl', OWL)
    g.bind('xsd', XSD)
    g.bind('mfg', mfg)
    g.bind('rr', rr)
    g.bind('sim', sim)
    
    # Add base ontology definitions
    for node, data in ontology.nodes(data=True):
        if data.get('type') == 'entity':
            # Create class definition
            class_uri = mfg[node]
            g.add((class_uri, RDF.type, OWL.Class))
            g.add((class_uri, RDFS.label, Literal(node)))
    
    # Add R2RML mappings
    for table, data in ontology.nodes(data=True):
        if data.get('type') == 'entity':
            mapping = BNode()
            g.add((URIRef(f'#Mapping_{table}'), rr.logicalTable, mapping))
            sql_query = f"""SELECT * FROM {table}"""
            g.add((mapping, rr.sqlQuery, Literal(sql_query)))
    
    # Add relationships and attributes
    for edge in ontology.edges(data=True):
        subject = mfg[edge[0]]
        predicate_type = edge[2].get('type', '')
        
        if predicate_type == 'has_attribute':
            # Create property for attribute
            property_uri = mfg[f'has{edge[1].capitalize()}']
            g.add((property_uri, RDF.type, OWL.DatatypeProperty))
            g.add((property_uri, RDFS.domain, subject))
        else:
            # Create object property for relationships
            property_uri = mfg[predicate_type]
            g.add((property_uri, RDF.type, OWL.ObjectProperty))
            g.add((property_uri, RDFS.domain, subject))
            g.add((property_uri, RDFS.range, mfg[edge[1]]))

   
    
    return g

def save_rdf_to_file(filename, rdf_graph):
    # Save with custom formatting
    turtle_str = rdf_graph.serialize(format='turtle')
    
    # Add section comments
    final_str = """# Base Ontology\n""" + \
               turtle_str + \
               """\n# R2RML Mappings\n""" + \
               """# Sample Data Instances\n"""
               
    with open(filename, 'w') as f:
        f.write(final_str)



# Modify the route to use the class
@app.route('/construct_ontology', methods=['POST'])
def construct_ontology():
    # Load database configuration and create engine
    db_config = load_db_config()
    connection_string = f"mysql+pymysql://{db_config['username']}:{db_config['password']}@{db_config['host']}/{db_config['dbname']}"
    engine = create_engine(connection_string)
    
    metadata = get_database_metadata()
    relationships = identify_relationships(metadata)
    ontology = build_local_ontology(metadata, relationships)
    rdf_graph = convert_to_rdf(ontology)
    
    # Save main ontology
    save_rdf_to_file('ontology2.ttl', rdf_graph)
    
    # Create resolver instance and perform entity resolution
    resolver = OntologyResolver()
    resolution_graph, metrics = resolver.check_entity_resolution(rdf_graph, engine)
    resolver.save_entity_resolution('entity_resolution.ttl', resolution_graph, metrics)
    
    # Convert ontology to vis.js format
    nodes = [{'id': node, 'label': node, 'group': 'entity' if data.get('type') == 'entity' else 'attribute'}
             for node, data in ontology.nodes(data=True)]
    edges = [{'from': edge[0], 'to': edge[1], 'label': ontology.edges[edge].get('type', '')}
             for edge in ontology.edges()]
    
    return jsonify({
        'nodes': nodes, 
        'edges': edges,
        'entity_resolution_metrics': metrics
    })
# Function to handle model generation request
def generate(payload):
    try:
        with requests.post('http://localhost:11434/api/generate', json=payload, stream=True) as response:
            response.raise_for_status()
            full_response = ""
            for line in response.iter_lines():
                if line:
                    json_response = json.loads(line)
                    if 'response' in json_response:
                        full_response += json_response['response']
                        yield json_response['response']
            
            return full_response
    
    except requests.RequestException as e:
        return f"Error: {str(e)}"

# Route to create prompt and send to model API
@app.route('/ask', methods=['GET'])
def ask_gemini():
    try:
        # Konfigurasi database
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="newbps"
        )
        cursor = conn.cursor(dictionary=True)

        # Get database metadata
        metadata = get_database_metadata()
        
        # Template awal ontology
        ontology = """@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix cs: <http://example.org/customersatisfaction#> .

# ============= CLASS DEFINITIONS =============
# Classes adalah template/blueprint untuk instances
"""
        
        # Add classes
        for table in metadata.keys():
            class_name = table.capitalize()
            ontology += f"""cs:{class_name} a owl:Class ;
    rdfs:label "{class_name}" ;
    rdfs:comment "A {class_name.lower()} in the system" .

"""
        
        ontology += """# ============= PROPERTY DEFINITIONS =============
# Object Properties (menghubungkan instances dengan instances lain)
"""
        
        # Add object properties
        for table, info in metadata.items():
            for field in info['fields']:
                if field.endswith('_id'):
                    related_class = field.replace('_id', '').capitalize()
                    property_name = f"has{related_class}"
                    ontology += f"""cs:{property_name} a owl:ObjectProperty ;
    rdfs:domain cs:{table.capitalize()} ;
    rdfs:range cs:{related_class} ;
    rdfs:label "has {related_class.lower()}" .

"""
        
        ontology += """# Data Properties (menghubungkan instances dengan nilai literal)
"""
        
        # Add data properties
        for table, info in metadata.items():
            for field in info['fields']:
                if not field.endswith('_id'):
                    ontology += f"""cs:{field} a owl:DatatypeProperty ;
    rdfs:domain cs:{table.capitalize()} ;
    rdfs:range xsd:string ;
    rdfs:label "{field.replace('_', ' ')}" .

"""
        
        ontology += """# ============= INSTANCE DATA =============
"""
        
        # Add instances from database
        for table, info in metadata.items():
            class_name = table.capitalize()
            ontology += f"# Instances dari {class_name} Class\n"
            
            # Query untuk mengambil data dari tabel
            cursor.execute(f"SELECT * FROM {table}")
            rows = cursor.fetchall()
            
            for row in rows:
                # Instance identifier menggunakan id dari database
                instance_id = row.get('id', 'unknown')
                ontology += f"cs:{class_name}_{instance_id} a cs:{class_name} ;    # Instance dari class {class_name}\n"
                
                # Add property values dari data database
                for field, value in row.items():
                    if field != 'id':  # Skip ID field
                        if field.endswith('_id'):
                            # Handle foreign key relationships
                            related_class = field.replace('_id', '').capitalize()
                            property_name = f"has{related_class}"
                            if value is not None:
                                ontology += f"    cs:{property_name} cs:{related_class}_{value} ;    # Menggunakan Object Property\n"
                        else:
                            # Handle regular fields
                            if value is not None:
                                # Escape special characters dan handle tipe data
                                if isinstance(value, str):
                                    value = value.replace('"', '\\"')  # Escape quotes
                                    ontology += f"    cs:{field} \"{value}\" ;    # Menggunakan Data Property\n"
                                elif isinstance(value, (int, float)):
                                    ontology += f"    cs:{field} \"{value}\"^^xsd:decimal ;    # Menggunakan Data Property\n"
                                elif isinstance(value, datetime.datetime):
                                    ontology += f"    cs:{field} \"{value.isoformat()}\"^^xsd:dateTime ;    # Menggunakan Data Property\n"
                                elif isinstance(value, bool):
                                    ontology += f"    cs:{field} \"{str(value).lower()}\"^^xsd:boolean ;    # Menggunakan Data Property\n"
                
                # Remove trailing semicolon and add period
                ontology = ontology.rstrip(' ;\n') + " .\n\n"
        
        # Close database connection
        cursor.close()
        conn.close()
        
        # Save the ontology to a file
        file_path = os.path.join(os.getcwd(), 'localontology.ttl')
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(ontology)
            
        return Response(
            f"Ontology has been saved to {file_path}",
            content_type='text/plain'
        )
        
    except Exception as e:
        return Response(
            f"Error generating ontology: {str(e)}", 
            content_type='text/plain',
            status=500
        )

# Fungsi helper untuk mendapatkan tipe data yang sesuai
def get_xsd_type(value):
    if isinstance(value, int):
        return "xsd:integer"
    elif isinstance(value, float):
        return "xsd:decimal"
    elif isinstance(value, datetime.datetime):
        return "xsd:dateTime"
    elif isinstance(value, bool):
        return "xsd:boolean"
    else:
        return "xsd:string"
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

# Flask route modification
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
        query1 = text(f"SELECT DISTINCT `{column1}` FROM `{table1}`")
        result1 = connection.execute(query1)
        values1 = [row[0] for row in result1]

        query2 = text(f"SELECT DISTINCT `{column2}` FROM `{table2}`")
        result2 = connection.execute(query2)
        values2 = [row[0] for row in result2]

    # Perform fuzzy matching
    matches = []
    threshold = 80  # Set a threshold for matching score

    for value1 in values1:
        best_match, score = process.extractOne(value1, values2, scorer=fuzz.token_sort_ratio)
        if score >= threshold:
            matches.append({
                "value1": value1,
                "value2": best_match,
                "score": score,
                "table1": table1,
                "table2": table2
            })

   # Generate explanation using Ollama
    explanation_prompt = f"""
    Analisis hasil entity resolution berikut dan berikan penjelasan yang jelas:

    - Membandingkan data antara tabel '{table1}' (kolom: {column1}) dengan tabel '{table2}' (kolom: {column2})
    - Ditemukan {len(matches)} kecocokan dengan skor kesamaan >= {threshold}%
    - Skor kecocokan tertinggi: {max([m['score'] for m in matches], default=0)}%
    - Skor kecocokan terendah: {min([m['score'] for m in matches], default=0)}%

    Tolong jelaskan apa arti hasil ini dan bagaimana implikasinya terhadap kualitas data. 
    Berikan juga rekomendasi untuk perbaikan data jika diperlukan.
    Jelaskan dengan bahasa yang mudah dipahami.
    """
    
    ollama_explanation = call_ollama_api(explanation_prompt)

    return jsonify({
        'data': matches,
        'total_matches': len(matches),
        'explanation': ollama_explanation
    })

@app.route('/entity_ontology')
def entity_ontology():
    ontology_data = ontology_builder.get_ontology_data() if os.path.exists("buildontology.ttl") else None
    return render_template('entityontology.html', data=ontology_data)

@app.route('/check_entity', methods=['GET'])
def ask_gemini2():
    try:
        # Konfigurasi Gemini API
        genai.configure(api_key='AIzaSyA1ZHwqVwzj1z9A8lWNhlXZfX5-sby__8A')
        model = genai.GenerativeModel('gemini-pro')
        
        # Membuat prompt dengan format yang lebih baik dan template output
        prompt = """lakukan entity resolution dari data in dan berikan output dengan format seperti template dibawah ini:

Template Output yang diinginkan:
@prefix cs: <http://example.org/cs#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
cs:Master_indikators_2 a cs:Master_indikators ;
    cs:id_mskeg "2"^^xsd:decimal ;
    cs:nama "Jumlah Penduduk Sensus Penduduk 2020" ;
    cs:konsep "\"Penduduk\"" ;
    owl:sameAs cs:Master_indikators_3 .    # Menandakan bahwa kedua entitas merujuk pada hal yang sama
# Tambahan metadata untuk memperjelas hubungan
cs:equivalence_reason a owl:Annotation ;
    cs:related_entity_1 cs:Master_indikators_2 ;
    cs:related_entity_2 cs:Master_indikators_3 ;
    cs:similarity_score "0.90"^^xsd:decimal ;
    cs:matching_fields "id_mskeg, definisi, produsen_data_codes, timestamps, status, submission_period" ;
    cs:resolution_date "2024-11-20"^^xsd:date .

Data Input untuk dianalisis:
cs:Master_indikators_2 a cs:Master_indikators ;
    cs:id_mskeg "2"^^xsd:decimal ;
    cs:nama "Jumlah Penduduk Sensus Penduduk 2020" ;
    cs:konsep "\"Penduduk\"" ;
    cs:definisi "Jumlah penduduk adalah ukuran absolut dari penduduk, dinyatakan dalam satuan jiwa. Penduduk Hasil Sensus Penduduk 2020 adalah penduduk menurut alamat domisili; yaitu semua orang (WNI dan WNA) yang tinggal di wilayah Negara Kesatuan Republik Indonesia selama satu tahun atau lebih dan atau mereka yang berdomisili kurang dari satu tahun tetapi bertujuan untuk menetap lebih dari satu tahun" ;
    cs:produsen_data_province_code "00" ;
    cs:produsen_data_city_code "0000" ;
    cs:last_sync "2023-10-31 09:06:59" ;
    cs:created_at "2021-07-18 17:00:00" ;
    cs:updated_at "2023-10-31 02:06:59" ;
    cs:status "APPROVED" ;
    cs:submission_period "2021" .

cs:Master_indikators_3 a cs:Master_indikators ;
    cs:id_mskeg "2"^^xsd:decimal ;
    cs:nama "Jumlah Penduduk Sensus Penduduk 2020 abc" ;
    cs:konsep "\"Penduduk\ abc"" ;
    cs:definisi "Jumlah penduduk adalah ukuran absolut dari penduduk, dinyatakan dalam satuan jiwa. Penduduk Hasil Sensus Penduduk 2020 adalah penduduk menurut alamat domisili; yaitu semua orang (WNI dan WNA) yang tinggal di wilayah Negara Kesatuan Republik Indonesia selama satu tahun atau lebih dan atau mereka yang berdomisili kurang dari satu tahun tetapi bertujuan untuk menetap lebih dari satu tahun" ;
    cs:produsen_data_province_code "00" ;
    cs:produsen_data_city_code "0000" ;
    cs:last_sync "2023-10-31 09:06:59" ;
    cs:created_at "2021-07-18 17:00:00" ;
    cs:updated_at "2023-10-31 02:06:59" ;
    cs:status "APPROVED" ;
    cs:submission_period "2021" .

cs:Master_indikators_4 a cs:Master_indikators ;
    cs:id_mskeg "3"^^xsd:decimal ;
    cs:nama "Tamu per kamar" ;
    cs:konsep "\"Tamu per kamar\"" ;
    cs:definisi "Tamu per kamar adalah rata-rata banyaknya tamu dalam satu kamar yang disewa yang dibedakan ke dalam tamu asing, domestik, asing dan domestic" ;
    cs:produsen_data_province_code "00" ;
    cs:produsen_data_city_code "0000" ;
    cs:last_sync "2023-10-31 09:07:00" ;
    cs:created_at "2020-12-28 17:00:00" ;
    cs:updated_at "2023-10-31 02:07:00" ;
    cs:status "APPROVED" ;
    cs:submission_period "2020" .

cs:Master_indikators_5 a cs:Master_indikators ;
    cs:id_mskeg "3"^^xsd:decimal ;
    cs:nama "Tamu per kamar 123" ;
    cs:konsep "\"Tamu per kamar 123\"" ;
    cs:definisi "Tamu per kamar adalah rata-rata banyaknya tamu dalam satu kamar yang disewa yang dibedakan ke dalam tamu asing, domestik, asing dan domestic" ;
    cs:produsen_data_province_code "00" ;
    cs:produsen_data_city_code "0000" ;
    cs:last_sync "2023-10-31 09:07:00" ;
    cs:created_at "2020-12-28 17:00:00" ;
    cs:updated_at "2023-10-31 02:07:00" ;
    cs:status "APPROVED" ;
    cs:submission_period "2020" .

cs:Master_indikators_6 a cs:Master_indikators ;
    cs:id_mskeg "3"^^xsd:decimal ;
    cs:nama "TPTT Hotel" ;
    cs:konsep "\"Tingkat Pemakaian Tempat Tidur (TPTT) Hotel\"" ;
    cs:definisi "Tingkat Pemakaian Tempat Tidur (TPTT) Hotel adalah perbandingan antara jumlah tempat tidur hotel yang telah dipakai dengan jumlah tempat tidur yang tersedia." ;
    cs:produsen_data_province_code "00" ;
    cs:produsen_data_city_code "0000" ;
    cs:last_sync "2023-10-31 09:07:00" ;
    cs:created_at "2020-12-28 17:00:00" ;
    cs:updated_at "2023-10-31 02:07:00" ;
    cs:status "APPROVED" ;
    cs:submission_period "2020" .

Tolong analisis data diatas dan berikan output sesuai template yang diberikan. Identifikasi entitas yang sama atau sangat mirip berdasarkan:
1. Kesamaan id_mskeg
2. Kemiripan nama dan konsep
3. Kemiripan definisi
4. Kesamaan kode produsen data
5. Kesamaan timestamps dan status
6. Kesamaan periode submission

Berikan similarity score berdasarkan berapa banyak field yang cocok dari kriteria diatas. Gunakan format yang sama persis dengan template, termasuk prefixes dan structure-nya."""
        
        # Generate konten
        response = model.generate_content(prompt)
        
        # Return plain text response
        return Response(response.text, content_type='text/plain')
            
    except Exception as e:
        return Response(f"Error: {str(e)}", content_type='text/plain')
    
@app.route('/check_gsbpm_entity', methods=['GET'])
def check_gsbpm_entity():
    try:
        # Konfigurasi Gemini API
        genai.configure(api_key='AIzaSyA1ZHwqVwzj1z9A8lWNhlXZfX5-sby__8A')
        model = genai.GenerativeModel('gemini-pro')
        
        # Load the GSBPM ontology file
        g = Graph()
        g.parse("gsbpm_ontology.ttl", format="turtle")
        
        def safe_n3(value):
            """Safely convert value to N3 format, handling None values"""
            if value is None:
                return '""'
            if isinstance(value, Literal):
                return value.n3()
            if isinstance(value, URIRef):
                return f"<{value}>"
            return f'"{str(value)}"'

        def extract_project_data(g):
            projects_data = ""
            GSBPM = Namespace("http://example.org/gsbpm#")
            
            # Query untuk mendapatkan semua Statistical Projects
            for project in g.subjects(RDF.type, GSBPM.StatisticalProject):
                project_id = project.split('#')[-1]
                projects_data += f"\ngsbpm:{project_id} a gsbpm:StatisticalProject ;\n"
                
                # Get basic properties
                basic_properties = {
                    "hasProjectID": None,
                    "hasProjectName": None,
                    "hasStartDate": None,
                    "statisticalType": None,
                    "dataCollectionMethod": None
                }
                
                for pred, obj in g.predicate_objects(project):
                    pred_name = pred.split('#')[-1]
                    if pred_name in basic_properties:
                        basic_properties[pred_name] = obj
                
                # Add non-null properties
                for prop_name, value in basic_properties.items():
                    if value is not None:
                        projects_data += f"    gsbpm:{prop_name} {safe_n3(value)} ;\n"
                
                # Get SpecifyNeeds phase data
                specify_needs = list(g.objects(project, GSBPM.hasSpecifyNeeds))
                if specify_needs:
                    needs_node = specify_needs[0]
                    projects_data += "    gsbpm:hasSpecifyNeeds [\n"
                    
                    # Get indicators
                    indicators = list(g.objects(needs_node, GSBPM.identifyNeeds))
                    if indicators:
                        projects_data += "        gsbpm:indicators [\n"
                        for indicator in indicators:
                            projects_data += f"            gsbpm:name {safe_n3(indicator)} ;\n"
                        projects_data += "        ] ;\n"
                    
                    projects_data += "    ] ;\n"
                
                # Get MetadataManagement data
                metadata = list(g.objects(project, GSBPM.hasMetadataManagement))
                if metadata:
                    metadata_node = metadata[0]
                    projects_data += "    gsbpm:hasMetadataManagement [\n"
                    
                    # Get variables
                    variables = list(g.objects(metadata_node, GSBPM.hasVariable))
                    if variables:
                        for variable in variables:
                            projects_data += f"        gsbpm:variable {safe_n3(variable)} ;\n"
                    
                    projects_data += "    ] .\n"
                else:
                    projects_data = projects_data.rstrip(" ;\n") + " .\n"
            
            return projects_data
        
        input_data = extract_project_data(g)
        
        if not input_data.strip():
            return "No valid projects found in the ontology", 404
        
        # Membuat prompt untuk analisis entity resolution
        prompt = f"""Lakukan entity resolution dari data GSBPM dan berikan output dengan format seperti template dibawah ini:

Template Output yang diinginkan:
@prefix gsbpm: <http://example.org/gsbpm#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

gsbpm:Project_1 a gsbpm:StatisticalProject ;
    gsbpm:hasProjectID "1"^^xsd:decimal ;
    gsbpm:hasProjectName "Census Project" ;
    owl:sameAs gsbpm:Project_2 .

gsbpm:equivalence_reason a owl:Annotation ;
    gsbpm:related_entity_1 gsbpm:Project_1 ;
    gsbpm:related_entity_2 gsbpm:Project_2 ;
    gsbpm:similarity_score "0.90"^^xsd:decimal ;
    gsbpm:matching_fields "projectID, projectName, indicators, metadata" ;
    gsbpm:resolution_date "2024-11-20"^^xsd:date .

Data Input untuk dianalisis:
{input_data}

Tolong analisis data diatas dan berikan output sesuai template yang diberikan. Identifikasi entitas yang sama atau sangat mirip berdasarkan:
1. Kesamaan projectID
2. Kemiripan projectName
3. Kemiripan dalam fase SpecifyNeeds (indicators)
4. Kemiripan dalam MetadataManagement
5. Kesamaan timestamps
6. Kesamaan phases yang terlibat

Berikan similarity score berdasarkan berapa banyak field yang cocok dari kriteria diatas. Gunakan format yang sama persis dengan template, termasuk prefixes dan structure-nya."""
        
        # Generate konten
        response = model.generate_content(prompt)
        
        # Return response dalam format Turtle
        return Response(
            response.text,
            mimetype='text/turtle',
            headers={
                'Content-Disposition': 'attachment; filename=gsbpm_entity_resolution.ttl'
            }
        )
            
    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}", 500
    
@app.route('/dashboard')
def entity_resolution_dashboard():
    try:
        # Read the Turtle output (mock for now, or read from a file)
        with open("gsbpm_entity_resolution.ttl", "r") as file:
            turtle_data = file.read()
        
        # Parse the Turtle file into JSON-LD or triples for frontend
        g = Graph()
        g.parse(data=turtle_data, format="turtle")
        
        GSBPM = Namespace("http://example.org/gsbpm#")
        OWL = Namespace("http://www.w3.org/2002/07/owl#")
        XSD = Namespace("http://www.w3.org/2001/XMLSchema#")
        
        # Extract relevant data
        entities = []
        for subject in g.subjects(RDF.type, GSBPM.StatisticalProject):
            project_data = {
                "project_id": g.value(subject, GSBPM.hasProjectID),
                "project_name": g.value(subject, GSBPM.hasProjectName),
                "same_as": list(g.objects(subject, OWL.sameAs)),
            }
            entities.append(project_data)
        
        resolutions = []
        for subject in g.subjects(RDF.type, OWL.Annotation):
            resolution = {
                "related_entity_1": g.value(subject, GSBPM.related_entity_1),
                "related_entity_2": g.value(subject, GSBPM.related_entity_2),
                "similarity_score": g.value(subject, GSBPM.similarity_score),
                "matching_fields": g.value(subject, GSBPM.matching_fields),
                "resolution_date": g.value(subject, GSBPM.resolution_date),
            }
            resolutions.append(resolution)
        
        # Render the dashboard with the extracted data
        return render_template(
            'dashboard.html',
            entities=entities,
            resolutions=resolutions
        )
    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}", 500

if __name__ == '__main__':
    app.run(debug=True)
