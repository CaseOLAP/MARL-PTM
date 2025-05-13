import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class ExpressionLoader:
    """
    Loads and processes gene expression data for PTM prediction.
    Supports bulk tissue or disease-specific expression from GTEx or TCGA.
    Produces PCA-compressed features for each gene/protein.
    """

    def __init__(self, expression_file, pca_dim=128):
        self.expression_file = expression_file
        self.pca_dim = pca_dim

    def load_expression_matrix(self):
        """
        Loads the raw gene expression matrix.
        Expected format: rows = genes, columns = samples.
        Values in TPM or FPKM.
        """
        df = pd.read_csv(self.expression_file, index_col=0)
        return df

    def preprocess_expression(self, df):
        """
        Applies log transformation and z-score normalization.
        Returns normalized matrix of shape (genes × samples)
        """
        log_df = np.log2(df + 1)
        scaler = StandardScaler()
        norm_data = scaler.fit_transform(log_df.T).T  # normalize per gene
        return pd.DataFrame(norm_data, index=log_df.index, columns=log_df.columns)

    def reduce_dimensionality(self, norm_df):
        """
        Applies PCA to reduce expression data to a fixed-length vector per gene.
        Output: {gene_id: PCA_embedding_vector}
        """
        pca = PCA(n_components=self.pca_dim)
        reduced = pca.fit_transform(norm_df.T)  # genes x pca_dim
        reduced_df = pd.DataFrame(reduced, index=norm_df.columns)

        gene_map = {}
        for gene in reduced_df.index:
            gene_map[gene] = reduced_df.loc[gene].values.astype(np.float32)
        return gene_map

    def load_embeddings(self):
        """
        Main entry point: loads, preprocesses, and reduces expression data.
        Returns: {gene_name: 128D expression vector}
        """
        raw_df = self.load_expression_matrix()
        norm_df = self.preprocess_expression(raw_df)
        return self.reduce_dimensionality(norm_df)
