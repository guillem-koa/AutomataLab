def detail2class(table4detail):
    import pandas as pd
    chunk_df_list = []
    for replica in [replica for replica in table4detail['Replica'].unique() if len(replica)>0]:
        for input in [input for input in table4detail['InputName'].unique() if len(input)>0]:
            
            filtered_df = table4detail[(table4detail['Replica']==replica) & (table4detail['InputName']==input)]

            if len(filtered_df)!=0: # We want to avoid empty rows (example: C5 has 3 replicas, and C2 only has 2)

                # Get input concentration (should be a unique value)
                if len(filtered_df['InputConcentration'].unique())==1:
                    concentration = filtered_df['InputConcentration'].unique()
                else: 
                    print("Cuidado, apareixen diferents concentracions amb diferents rèpliques!")

                filtered_df = filtered_df[['ReporterName', 'gfp']].T
                filtered_df.columns = filtered_df.iloc[0]
                filtered_df = filtered_df[1:]
                filtered_df['Replica'] = replica
                filtered_df['InputName'] = input
                filtered_df['InputConcentration'] = concentration

                chunk_df_list.append(filtered_df)

    table4class = pd.concat(chunk_df_list)
    return table4class