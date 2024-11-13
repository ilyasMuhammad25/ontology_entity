# entity_ontology.py
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS, OWL, XSD
from sqlalchemy import create_engine, text

class OntologyBuilder:
    def __init__(self):
        self.MFG = Namespace("http://example.org/manufacturing#")
        self.RR = Namespace("http://www.w3.org/ns/r2rml#")

    def database_to_ontology(self, metadata, db_config):
        # Create database connection
        connection_string = f"mysql+pymysql://{db_config['username']}:{db_config['password']}@{db_config['host']}/{db_config['dbname']}"
        engine = create_engine(connection_string)
        
        # Create a new RDF graph
        g = Graph()
        
        # Bind namespaces
        g.bind("rdf", RDF)
        g.bind("rdfs", RDFS)
        g.bind("owl", OWL)
        g.bind("xsd", XSD)
        g.bind("mfg", self.MFG)
        g.bind("rr", self.RR)
        
        # Create ontology header
        ontology = URIRef("http://example.org/manufacturing")
        g.add((ontology, RDF.type, OWL.Ontology))
        
        # Process each table
        with engine.connect() as connection:
            for table_name, table_info in metadata.items():
                # Create class for table
                class_uri = self.MFG[table_name.capitalize()]
                g.add((class_uri, RDF.type, OWL.Class))
                g.add((class_uri, RDFS.label, Literal(table_name)))
                
                # Create properties for columns
                for field in table_info['fields']:
                    prop_uri = self.MFG[f"has{field.capitalize()}"]
                    g.add((prop_uri, RDF.type, OWL.DatatypeProperty))
                    g.add((prop_uri, RDFS.domain, class_uri))
                    g.add((prop_uri, RDFS.label, Literal(field)))
                    g.add((prop_uri, RDFS.range, XSD.string))
                
                # Get actual data from database
                query = f"SELECT * FROM {table_name}"
                try:
                    result = connection.execute(text(query))
                    rows = result.fetchall()
                    
                    # Add R2RML mapping with actual query
                    mapping_uri = URIRef(f"#Mapping_{table_name}")
                    g.add((mapping_uri, self.RR.logicalTable, URIRef(f"#LogicalTable_{table_name}")))
                    g.add((URIRef(f"#LogicalTable_{table_name}"), self.RR.sqlQuery, Literal(query)))
                    
                    # Create instances for each row
                    for idx, row in enumerate(rows):
                        instance_uri = self.MFG[f"{table_name}_{idx+1}"]
                        g.add((instance_uri, RDF.type, class_uri))
                        
                        # Add properties for each column
                        for field, value in zip(table_info['fields'], row):
                            if value is not None:  # Skip null values
                                prop_uri = self.MFG[f"has{field.capitalize()}"]
                                g.add((instance_uri, prop_uri, Literal(str(value))))
                                
                except Exception as e:
                    print(f"Error querying table {table_name}: {str(e)}")
        
        # Save to file
        output_file = "buildontology.ttl"
        g.serialize(destination=output_file, format="turtle")
        return output_file

    def get_ontology_data(self):
        try:
            g = Graph()
            g.parse("buildontology.ttl", format="turtle")
            
            classes = {}
            # Get all classes
            for s, p, o in g.triples((None, None, None)):
                if str(p) == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" and \
                   str(o) == "http://www.w3.org/2002/07/owl#Class":
                    class_name = str(s).split('#')[-1]
                    classes[class_name] = {
                        'instances': [],
                        'properties': []
                    }
            
            # Get properties and instances for each class
            for class_name in classes:
                class_uri = self.MFG[class_name]
                
                # Get properties
                for s, p, o in g.triples((None, None, class_uri)):
                    if str(p) == "http://www.w3.org/2000/01/rdf-schema#domain":
                        prop_name = str(s).split('#')[-1]
                        classes[class_name]['properties'].append(prop_name)
                
                # Get instances
                for s, p, o in g.triples((None, None, class_uri)):
                    if str(p) == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type":
                        instance_uri = str(s)
                        instance = {'uri': instance_uri, 'properties': {}}
                        
                        # Get instance properties
                        for ps, pp, po in g.triples((s, None, None)):
                            if str(pp).startswith(str(self.MFG)):
                                prop_name = str(pp).split('#')[-1]
                                instance['properties'][prop_name] = str(po)
                        
                        classes[class_name]['instances'].append(instance)
            
            return classes
        except Exception as e:
            print(f"Error reading ontology: {str(e)}")
            return None