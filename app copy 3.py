from flask import Flask, jsonify,request,Response,render_template, request
from rdflib import Graph, Namespace, Literal, URIRef, BNode
from rdflib.namespace import RDF, RDFS, OWL, XSD
import networkx as nx
import requests
from difflib import SequenceMatcher
import re
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



class OntologyResolver:
    def __init__(self):
        self.mfg = Namespace("http://example.org/manufacturing#")
        
    def get_table_schema(self, table_name, engine):
        """Get column information for a table"""
        inspector = inspect(engine)
        return inspector.get_columns(table_name)
        
    def get_table_sample_data(self, table_name, engine, sample_size=1000):
        """Get sample data from table"""
        query = f"SELECT * FROM {table_name} LIMIT {sample_size}"
        return pd.read_sql(query, engine)

    def compare_columns(self, cols1, cols2):
        """Compare two sets of columns and return similarity metrics"""
        # Get column names
        names1 = set(col['name'] for col in cols1)
        names2 = set(col['name'] for col in cols2)
        
        # Calculate overlapping columns
        common_cols = names1.intersection(names2)
        
        # Calculate type matches for common columns
        type_matches = sum(1 for col in cols1 if col['name'] in common_cols 
                         and any(col2['type'].__str__() == col['type'].__str__() 
                               for col2 in cols2 if col2['name'] == col['name']))
        
        metrics = {
            'column_overlap': len(common_cols) / max(len(names1), len(names2)),
            'type_match_ratio': type_matches / len(common_cols) if common_cols else 0,
            'common_columns': list(common_cols)
        }
        
        return metrics

    def compare_data_patterns(self, data1, data2, common_cols):
        """Compare data patterns between two tables for common columns"""
        pattern_metrics = {}
        
        for col in common_cols:
            if col in data1.columns and col in data2.columns:
                # Compare basic statistics
                stats1 = data1[col].describe()
                stats2 = data2[col].describe()
                
                # Calculate correlation if numeric
                if pd.api.types.is_numeric_dtype(data1[col]) and pd.api.types.is_numeric_dtype(data2[col]):
                    correlation = data1[col].corr(data2[col])
                    pattern_metrics[col] = {
                        'correlation': correlation,
                        'mean_diff': abs(stats1['mean'] - stats2['mean']),
                        'std_diff': abs(stats1['std'] - stats2['std'])
                    }
                else:
                    # For non-numeric, compare value distributions
                    dist1 = data1[col].value_counts(normalize=True)
                    dist2 = data2[col].value_counts(normalize=True)
                    
                    # Calculate Jensen-Shannon divergence or similar metric
                    pattern_metrics[col] = {
                        'value_overlap': len(set(data1[col]).intersection(set(data2[col]))) / \
                                       len(set(data1[col]).union(set(data2[col])))
                    }
        
        return pattern_metrics

    def check_entity_resolution(self, rdf_graph, engine):
        """
        Enhanced entity resolution with schema and data analysis
        """
        resolution_graph = Graph()
        resolution_graph.bind('mfg', self.mfg)
        resolution_graph.bind('owl', OWL)
        resolution_graph.bind('rdfs', RDFS)
        
        # Extract all classes
        classes = []
        for s, p, o in rdf_graph.triples((None, RDF.type, OWL.Class)):
            classes.append(str(s))
        
        similar_entities = []
        similarity_scores = []
        
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                class1 = classes[i].split('#')[1]
                class2 = classes[j].split('#')[1]
                
                # Basic name similarity
                base1 = re.sub(r'_copy\d+$', '', class1)
                base2 = re.sub(r'_copy\d+$', '', class2)
                name_sim_score = SequenceMatcher(None, base1, base2).ratio()
                
                # Only proceed with detailed analysis if names are similar enough
                if base1 == base2 or name_sim_score > 0.9:
                    # Get schema information
                    cols1 = self.get_table_schema(class1, engine)
                    cols2 = self.get_table_schema(class2, engine)
                    column_metrics = self.compare_columns(cols1, cols2)
                    
                    # Get sample data for detailed comparison
                    data1 = self.get_table_sample_data(class1, engine)
                    data2 = self.get_table_sample_data(class2, engine)
                    data_metrics = self.compare_data_patterns(data1, data2, column_metrics['common_columns'])
                    
                    # Calculate overall similarity score
                    overall_score = (
                        name_sim_score * 0.3 +  # Name similarity weight
                        column_metrics['column_overlap'] * 0.3 +  # Schema similarity weight
                        column_metrics['type_match_ratio'] * 0.4   # Data pattern similarity weight
                    )
                    
                    similarity_info = {
                        'class1': class1,
                        'class2': class2,
                        'name_similarity': name_sim_score,
                        'column_metrics': column_metrics,
                        'data_patterns': data_metrics,
                        'overall_score': overall_score
                    }
                    
                    similarity_scores.append(similarity_info)
                    
                    if overall_score > 0.8:  # Threshold for considering tables as similar
                        similar_entities.append((classes[i], classes[j]))
                        
                        uri1 = URIRef(classes[i])
                        uri2 = URIRef(classes[j])
                        
                        # Add to resolution graph
                        resolution_graph.add((uri1, RDF.type, OWL.Class))
                        resolution_graph.add((uri2, RDF.type, OWL.Class))
                        resolution_graph.add((uri1, OWL.equivalentClass, uri2))
                        
                        # Add detailed similarity metrics as annotations
                        comment = f"""Similarity Analysis:
                            Name Similarity: {name_sim_score:.2f}
                            Column Overlap: {column_metrics['column_overlap']:.2f}
                            Type Match Ratio: {column_metrics['type_match_ratio']:.2f}
                            Overall Score: {overall_score:.2f}"""
                        
                        resolution_graph.add((uri1, RDFS.comment, Literal(comment)))
        
        metrics = {
            'total_classes': len(classes),
            'similar_pairs_found': len(similar_entities),
            'similarity_scores': similarity_scores
        }
        
        return resolution_graph, metrics

    def save_entity_resolution(self, filename, resolution_graph, metrics):
        """Save enhanced resolution results with detailed metrics"""
        header = f"""# Entity Resolution Results
# Total Classes: {metrics['total_classes']}
# Similar Pairs Found: {metrics['similar_pairs_found']}
# Detailed Similarity Scores:
"""
        
        for score in metrics['similarity_scores']:
            header += f"""
# {score['class1']} - {score['class2']}:
#   Name Similarity: {score['name_similarity']:.2f}
#   Column Overlap: {score['column_metrics']['column_overlap']:.2f}
#   Type Match Ratio: {score['column_metrics']['type_match_ratio']:.2f}
#   Overall Score: {score['overall_score']:.2f}
#   Common Columns: {', '.join(score['column_metrics']['common_columns'])}
"""
        
        turtle_str = resolution_graph.serialize(format='turtle')
        final_str = header + "\n" + turtle_str
        
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


if __name__ == '__main__':
    app.run(debug=True)
