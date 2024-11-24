@prefix cs: <http://example.org/cs#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

cs:Master_indikators_2 a cs:Master_indikators ;
    cs:id_mskeg "2"^^xsd:decimal ;
    cs:nama "Jumlah Penduduk Sensus Penduduk 2020" ;
    cs:konsep ""Penduduk"" ;
    owl:sameAs cs:Master_indikators_3 .    # Menandakan bahwa kedua entitas merujuk pada hal yang sama

# Tambahan metadata untuk memperjelas hubungan
cs:equivalence_reason a owl:Annotation ;
    cs:related_entity_1 cs:Master_indikators_2 ;
    cs:related_entity_2 cs:Master_indikators_3 ;
    cs:similarity_score "0.90"^^xsd:decimal ;
    cs:matching_fields "id_mskeg, nama, konsep, definisi, produsen_data_province_code, produsen_data_city_code, timestamps, status, submission_period" ;
    cs:resolution_date "2024-11-20"^^xsd:date .