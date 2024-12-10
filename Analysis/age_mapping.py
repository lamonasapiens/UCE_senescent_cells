'''This file contains the necessary items to plot the percentage of senescent cells vs age'''

age_mapping = {
    'bm1_B1': 28,
    'bm1_B2': 46,
    'bm1_B3': 24,
    'bm1_B4': 55,
    'bm1_B5': 56,
    'bm1_B6': 55,
    'bm1_B7': 56,
    'bm1_B8': 31,
    'bm1_B9': 66,
    'bm1_B10': 58,
    'bm1_B11': 50,
    'bm1_B12': 67,
    'bm1_B13': 60,
    'bm1_B14': 57,
    'bm1_B15': 84,
    'bm1_B16': 43,
    'bm1_B17': 50,
    'bm1_B18': 58,
    'bm1_B19': 41,
    'bm1_B20': 30,
    'bm1_B21': 60,
    'bm1_B22': 59,
    'bm1_B23': 60,
    'bm1_B24': 47,
    'bm1_B25': 59,
    'sk1_S1': 29,
    'sk1_S2': 71,
    'sk1_S3': 70,
    'sk1_S4': 24,
    'sk1_S5': 67,
    'sk1_S6': 66,
    'sk1_S7': 24,
    'sk1_S8': 29
}


def get_age_plots(adata, tissue):
    '''This function plots the percentage of senescent cells by age in a given tissue'''
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # Drop cells with NaN ages and select only bone marrow samples
    adata_filtered = adata[adata.obs['tissue'] == tissue].copy()
    adata_filtered = adata_filtered[~adata_filtered.obs['age'].isna()].copy()

    # Group by age and Sen_consensus, then count
    age_distribution = adata_filtered.obs.groupby(['age', 'Sen_consensus']).size().reset_index(name='count')

    # Calculate total cells for each age group
    total_cells_per_age = age_distribution.groupby('age')['count'].transform('sum')

    # Calculate percentages of senescent cells for each age
    age_distribution['percentage'] = (age_distribution['count'] / total_cells_per_age) * 100

    # Bar plot: percentage of senescent cells vs age
    senescent_data = age_distribution[age_distribution['Sen_consensus'] == 'senescent']

    # Bar plot for senescent cells percentage
    plt.figure(figsize=(10, 6))
    sns.barplot(data=senescent_data, x='age', y='percentage', color='red') 
    plt.title(f'Percentage of Senescent Cells by Age in {tissue}')
    plt.xlabel('Age')
    plt.ylabel('Percentage of Senescent Cells')
    plt.tight_layout() 
    plt.show()
