class utils():
        '''
        A util
        
        Parameters
        ------------------------------------------------------------------------------------------------
        Name: str
                 save Name or data Name
        Returns
        ------------------------------------------------------------------------------------------------
        filename: str
                 file name    
        Examples
        -------------------------------------------------------------------------------------------------
        >>> filename = utils.saveName_transfer_to_filename('Krebs_Cycle_16Nodes_43Edges_TS')
        >>> print(filename)
        >>> filename = utils.dataName_transfer_to_filename('Krebs_Cycle_16Nodes_43Edges_TS_100Datasize_50Timesets')
        >>> print(filename)
        '''
  
    def __init__(self, filename):
        self.filename = filename

    def saveName_transfer_to_filename(filename):
        parts = filename.split('_')
        name_without_extension = parts[:-1]
        parts_to_join = name_without_extension[:-2]
        
        # join with '_'
        result = '_'.join(parts_to_join)
        return result

    def dataName_transfer_to_filename(filename):
        parts = filename.split('_')
        name_without_extension = parts[:-1]
        parts_to_join = name_without_extension[:-4]
        
        # join with '_'
        result = '_'.join(parts_to_join)
        return result

    def readable_File(FilePATH):
        read_Dir=os.listdir(FilePATH)
        count = 0
        readable_F = []
        for f in read_Dir:
            file = os.path.join(FilePATH, f)
            if os.path.isdir(file):
                count = count+1
            else:
                readable_F.append(f)
        return count,readable_F
