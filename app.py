from flask import Flask, jsonify,request,Response,render_template, request
from flask import Response, redirect, url_for
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
        database="bps_new"
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

# untuk mengenerate ontology local
@app.route('/gsbpm-ontology')
def get_gsbpm_ontology():
    """
    Generate GSBPM ontology and save it as TTL file in the app directory
    """
    try:
        # Create the ontology graph
        g = create_gsbpm_ontology()
        
        # Get the directory where the Flask app is located
        app_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(app_dir, 'gsbpm_ontology.ttl')
        
        # Serialize and save to file
        g.serialize(destination=file_path, format='turtle')
        
        return Response(
            f"File successfully generated at {file_path}",
            mimetype='text/plain'
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

# melakukan check entity resolution
@app.route('/check_gsbpm_entity', methods=['GET'])
def check_gsbpm_entity():
    try:
        # Konfigurasi Gemini API
        genai.configure(api_key='AIzaSyA1ZHwqVwzj1z9A8lWNhlXZfX5-sby__8A')
        model = genai.GenerativeModel('gemini-pro')
        
        # Get the directory where the Flask app is located
        app_dir = os.path.dirname(os.path.abspath(__file__))
        ontology_path = os.path.join(app_dir, 'gsbpm_ontology.ttl')
        # cek file gsbpm_ontology.ttl
        
        # Load the GSBPM ontology file
        g = Graph()
        g.parse(ontology_path, format="turtle")
        
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

gsbpm:equivalence_reason_1 a owl:Annotation ;
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
        
        # Save response to file
       # Save response to file
        output_path = os.path.join(app_dir, 'gsbpm_entity_resolution.ttl')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        # Instead of returning a Response, redirect to dashboard
        return redirect(url_for('entity_resolution_dashboard'))
            
    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}", 500
            
    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}", 500
    
# untuk menampilkan outputnya    
@app.route('/dashboard')
def entity_resolution_dashboard():
    try:
        # Read the Turtle output (mock for now, or read from a file)
        with open("gsbpm_entity_resolution.ttl", "r") as file:
            ttl_content = file.read() 
        
        # Parse the Turtle file into JSON-LD or triples for frontend
        g = Graph()
        g.parse(data=ttl_content, format="turtle")
        
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
            ttl_content=ttl_content,  # Pass the TTL content
            entities=entities,
            resolutions=resolutions
        )
    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}", 500
    




if __name__ == '__main__':
    app.run(debug=True)
