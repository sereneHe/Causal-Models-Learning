# Test  Two Dimensions Causality Data
if __name__ == "__main__":
    ############################################################################################################
    ############################################ SETTING File_PATH and file_name ###############################
    ############################################################################################################

    File_PATH = "./Test_Datasets/Real_data/"
    file_name = 'linearGauss_6_15'
    dt = ANMPOP_ReadData(File_PATH, file_name)
    dt.Produce_Rawdata()

# Test Three Dimensions Causality Time Series Data
if __name__ == "__main__":
    ############################################################################################################
    ############################################ SETTING File_PATH and file_name ###############################
    ############################################################################################################

    File_PATH = "./Test_Datasets/Real_data/"
    file_name = 'Krebs_Cycle'
    dt = ANMPOP_ReadData(File_PATH, file_name)
    dt.Produce_Rawdata()
