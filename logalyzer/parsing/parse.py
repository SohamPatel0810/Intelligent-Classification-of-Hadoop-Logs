from logparser.Drain import LogParser
import os
import pandas as pd

class HadoopLogParser:
    def __init__(self, log_format, input_dir, output_dir, depth=4, st=0.5, maxChild=100, regex=[], keep_para=False, output_file='merged_data.parquet'):
        self.log_format = log_format
        self.depth = depth
        self.st = st
        self.maxChild = maxChild
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.keep_para = False
        self.additional_regex = regex
        self.output_file = output_file
    
    def parse(self, log_file):
        parser = LogParser(log_format=self.log_format, indir=self.input_dir, outdir=self.output_dir , depth=self.depth, st=self.st, maxChild=self.maxChild, keep_para=self.keep_para, rex=self.additional_regex)
        parser.parse(log_file)
    
    def _get_csv_files(self, directory, extension='.log_structured.csv'):
        """List all files in a directory with a given extension."""
        return [file for file in os.listdir(directory) if file.endswith(extension)]

    def _read_and_append_csv(self, file_path, application_id):
        """Read a CSV file and add an application_id column."""
        df = pd.read_csv(file_path)   
        df['file_name'] = os.path.basename(file_path).split('.')[0]
        df['application_id'] = application_id
        return df

    def process_directory(self, base_directory):
        """Process all folders in the base directory and merge CSV files."""
        df_list = []
        for folder_name in os.listdir(base_directory):
            folder_path = os.path.join(base_directory, folder_name)
            if os.path.isdir(folder_path):
                csv_files = self._get_csv_files(folder_path)
                for file in csv_files:
                    file_path = os.path.join(folder_path, file)
                    df = self._read_and_append_csv(file_path, folder_name)
                    df_list.append(df)
        return df_list

    def merge_parsed_logs(self):
        df_list = self.process_directory(self.output_dir)
        merged_df = pd.concat(df_list, ignore_index=True)
        print('Merged dataframe shape:', merged_df.shape)
        
        # Save the merged dataframe to a directory as a parquet file
        merged_df.to_parquet(os.path.join(self.output_dir, self.output_file))
        