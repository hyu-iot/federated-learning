import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os

class Visualization:
    def __init__(self, config, path):
        self.config = config
        self.path = path

    def run(self):
        # result_path = os.path.join(self.path, 'result.csv')
        result_path = '/home/dongha/project/learning/result/temp_result_total.csv' #TODO: FIX THIS.
        df_read = pd.read_csv(filepath_or_buffer=result_path)

        appended_df = pd.DataFrame()
        for model in self.config.visualization['model']:
            for strat in self.config.visualization['strategy']:
                rslt_df = df_read[df_read['model'].str.match(model) & df_read['strategy'].str.match(strat)]
                appended_df = pd.concat([appended_df, rslt_df], axis=0)

        appended_df["name"] = df_read["model"] + "_" + df_read["strategy"]
        df_read.drop(columns=['model', 'strategy'], inplace=True)

        self.save_fig(appended_df)

        
    def save_fig(self, dataframe):
        
        rounds = sorted(dataframe['round'].unique())
        df_loss = pd.DataFrame(index=rounds)
        df_accuracy = pd.DataFrame(index=rounds)
        df_recall = pd.DataFrame(index=rounds)
        df_precision = pd.DataFrame(index=rounds)
        df_f1 = pd.DataFrame(index=rounds)

        df_result = {
            'loss' : df_loss, 
            'accuracy' : df_accuracy, 
            'recall' : df_recall, 
            'precision' : df_precision, 
            'f1-score' : df_f1}
        
        for name in dataframe['name'].unique():
            df_temp = dataframe[dataframe['name'] == name].sort_values(by=['round'])
            df_loss[name] = df_temp['c_loss'].values
            df_accuracy[name] = df_temp['c_accuracy'].values
            df_recall[name] = df_temp['c_recall'].values
            df_precision[name] = df_temp['c_precision'].values
            df_f1[name] = df_temp['c_f1_score'].values

        for key in self.config.visualization['metrics']:
            fig, ax = plt.subplots()
            for name in df_f1.columns:
                ax.plot(
                    rounds, 
                    df_result[key][name].values,
                    label=name,
                )

            ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            ax.set_xlabel("Number of Rounds", fontsize=16)
            ax.set_ylabel(key, fontsize=16)

            fig.set_dpi(300)
            fig.savefig(self.path+"/"+key+"_plot.png", bbox_inches='tight')