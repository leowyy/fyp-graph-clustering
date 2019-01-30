def main():
    from core.EmbeddingDataSet import EmbeddingDataSet
    dataset_name = 'pubmed'
    input_dir = '/Users/signapoop/Desktop/data'
    dataset = EmbeddingDataSet(dataset_name, input_dir)
    dataset.create_all_data(n_batches=50, shuffle=True, sampling=True)
    dataset.summarise()
    print([len(G.labels) for G in dataset.all_data])


if __name__ == '__main__':
    main()