import networkx as nx
import requests, pickle
import re
import pickle
import gzip
from urllib.parse import urlparse
import matplotlib.pyplot as plt


# init graph and mock dataset
g = nx.Graph()

url_data = [
    {"url": "http://username:password@www.example.com/path/to/resource?query=example&source=browser"},
    {"url": "https://phishy-site.com/login?user=test&pass=1234"},
    {"url" : "http://antitrust.altervista.org/wwww.popularenlinea.com.do"},
    {"url" : "http://antitrust.altervista.org/wwww.testing.com.do?query=example&source=browser"}
]


""" 
This function segments URL into words
    - removes stop words based on length
    - removes special characters from query string
    - splist domain into components

Returns:
    A set of words used in the URL

"""
def segment_url(url):

    words = set()
    parsed = urlparse(url)
    
    # Segment the hostname
    if parsed.hostname:
        words.update(parsed.hostname.split('.'))

    # Segment the path using punctuation symbols
    if parsed.path:
        words.update(re.split(r'[\/.!&,#$%;&]', parsed.path))

    # Segment the query string
    if parsed.query:
        queries = parsed.query.split('&')
        for query in queries:
            key_value = query.split('=')
            if len(key_value) == 2:
                words.add(key_value[0])
                words.add(key_value[1])

    # Remove empty strings and stop words
    words = {word for word in words if word and len(word) > 1}  
    print(words)
    return words

"""
This function to adds nodes and edges as defined in research paper:
edges between
    - URL -> DOMAIN
    - URL -> WORDS
    - DOMAIN -> authoritative name servers.

Returns:
    N/A
"""
def add_to_graph(url_entry):
    url = url_entry["url"] # fetch url
    parsed = urlparse(url) # parse the url

    # Create URL node
    url_node = f"URL: {url}"
    g.add_node(url_node, type="URL")

    # adding edge from URL to DOMAIN
    if parsed.hostname:
        domain_node = f"Domain:{parsed.hostname}"

        # Check if the domain node already exists in the graph, and add if not exist else reuse the domain
        if not g.has_node(domain_node):
            g.add_node(domain_node, type="Domain")  

        g.add_edge(url_node, domain_node)

    #  Draw an edge between a URL (i.e., sentence) and a substring (i.e., word)
    # Segment URL into words and create nodes for each word
    words = segment_url(url)
    for word in words:
        word_node = f"Word:{word}"
        g.add_node(word_node, type="Word")
        g.add_edge(url_node, word_node)



# Create graph .....
for entry in url_data:
    add_to_graph(entry)

# visualise the graph plot
def visualise_graph(graph):
    pos = nx.spring_layout(graph)  # Define layout
    plt.figure(figsize=(16, 16))

    # Draw nodes with different colors based on their type
    url_nodes = [n for n, attr in graph.nodes(data=True) if attr.get('type') == 'URL']
    domain_nodes = [n for n, attr in graph.nodes(data=True) if attr.get('type') == 'Domain']
    word_nodes = [n for n, attr in graph.nodes(data=True) if attr.get('type') == 'Word']

    nx.draw_networkx_nodes(graph, pos, nodelist=url_nodes, node_size=500, node_color="skyblue", label="URLs")
    nx.draw_networkx_nodes(graph, pos, nodelist=domain_nodes, node_size=500, node_color="lightgreen", label="Domains")
    nx.draw_networkx_nodes(graph, pos, nodelist=word_nodes, node_size=500, node_color="salmon", label="Words")

    # Draw edges and labels
    nx.draw_networkx_edges(graph, pos, edge_color="gray", alpha=0.5)
    nx.draw_networkx_labels(graph, pos, font_size=10, font_color="black")

    plt.title("Graph Visualization")
    plt.legend(scatterpoints=1, loc="upper right", fontsize=10)
    plt.show()

# Call the visualization function
visualise_graph(g)

# save graph to data/
with gzip.open('phishing_url/data/graph.gzpickle', 'wb') as f:
    pickle.dump(g, f)

print("Graph construction complete.")

