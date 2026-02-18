"""
Análisis de Redes Logísticas - Olist Logistics
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NetworkAnalyzer:
    """Sistema de análisis de redes logísticas"""
    
    def __init__(self, df_orders, df_sellers, df_items, df_customers):
        self.df_orders = df_orders
        self.df_sellers = df_sellers
        self.df_items = df_items
        self.df_customers = df_customers
        self.graph = None
        self.df_edges = None
        self.metrics = {}
        logger.info("NetworkAnalyzer inicializado")
    
    def build_graph(self):
        """Construye grafo dirigido de la red logística"""
        logger.info("Construyendo red logística...")
        
        df_route = self.df_items.merge(
            self.df_sellers[['seller_id', 'seller_city', 'seller_state']], 
            on='seller_id'
        )
        
        df_route = df_route.merge(
            self.df_orders[['order_id', 'customer_id']], 
            on='order_id'
        )
        
        df_route = df_route.merge(
            self.df_customers[['customer_id', 'customer_city', 'customer_state']], 
            on='customer_id'
        )
        
        self.df_edges = df_route.groupby(
            ['seller_city', 'customer_city']
        ).size().reset_index(name='weight')
        
        self.df_edges['seller_city'] = self.df_edges['seller_city'].str.title()
        self.df_edges['customer_city'] = self.df_edges['customer_city'].str.title()
        
        self.graph = nx.DiGraph()
        
        for _, row in self.df_edges.iterrows():
            self.graph.add_edge(row['seller_city'], row['customer_city'], weight=row['weight'])
        
        logger.info(f"   Nodos: {self.graph.number_of_nodes():,}")
        logger.info(f"   Aristas: {self.graph.number_of_edges():,}")
        
        return self.graph
    
    def calculate_centrality(self, top_n: int = 100):
        """Calcula métricas de centralidad"""
        logger.info("Calculando métricas...")
        
        in_degree = nx.in_degree_centrality(self.graph)
        out_degree = nx.out_degree_centrality(self.graph)
        
        top_nodes = sorted(self.graph.degree, key=lambda x: x[1], reverse=True)[:top_n]
        top_node_names = [n for n, d in top_nodes]
        subgraph = self.graph.subgraph(top_node_names)
        
        betweenness = nx.betweenness_centrality(subgraph, weight='weight')
        
        self.metrics = {'in_degree': in_degree, 'out_degree': out_degree, 'betweenness': betweenness}
        logger.info("✅ Métricas calculadas")
        
        return self.metrics
    
    def get_top_nodes(self, metric: str = 'out_degree', n: int = 10):
        """Obtiene top N nodos"""
        data = self.metrics[metric]
        sorted_data = sorted(data.items(), key=lambda x: x[1], reverse=True)[:n]
        df = pd.DataFrame(sorted_data, columns=['city', 'score'])
        df['rank'] = range(1, len(df) + 1)
        return df[['rank', 'city', 'score']]
    
    def get_vulnerability_report(self):
        """Genera reporte de vulnerabilidad"""
        if not self.metrics:
            self.calculate_centrality()
        
        report = {
            'top_producers': self.get_top_nodes('out_degree', 5),
            'top_consumers': self.get_top_nodes('in_degree', 5),
            'top_bridges': self.get_top_nodes('betweenness', 5),
            'sp_dominance': self.metrics['out_degree'].get('Sao Paulo', 0)
        }
        
        sp = report['sp_dominance']
        report['sp_risk'] = "CRÍTICO" if sp > 0.1 else "MEDIO" if sp > 0.05 else "BAJO"
        
        return report
    
    def print_report(self, report=None):
        """Imprime reporte formateado"""
        if report is None:
            report = self.get_vulnerability_report()
        
        print("="*70)
        print("REPORTE DE VULNERABILIDAD DE RED")
        print("="*70)
        
        print("\n🏭 TOP 5 PRODUCTORES:")
        for _, row in report['top_producers'].iterrows():
            print(f"   {row['rank']}. {row['city']}: {row['score']:.1%}")
        
        print("\n🛒 TOP 5 CONSUMIDORES:")
        for _, row in report['top_consumers'].iterrows():
            print(f"   {row['rank']}. {row['city']}: {row['score']:.1%}")
        
        print("\n🌉 TOP 5 PUENTES:")
        for _, row in report['top_bridges'].iterrows():
            print(f"   {row['rank']}. {row['city']}: {row['score']:.2%}")
        
        print(f"\n⚠️ DEPENDENCIA SÃO PAULO: {report['sp_dominance']:.1%}")
        print(f"   Riesgo: {report['sp_risk']}")
        print("="*70)
    
    def visualize_top_routes(self, n: int = 10):
        """Visualiza las rutas más importantes"""
        top = self.df_edges.nlargest(n, 'weight')
        
        print(f"\n🛣️ TOP {n} RUTAS:")
        print("-"*70)
        for _, row in top.iterrows():
            print(f"{row['seller_city']:25s} → {row['customer_city']:25s} | {row['weight']:,}")
        
        return top
