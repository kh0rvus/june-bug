import hyper_params

def collect (server):
    """
    collects data by continually requesting samples from the server
    and storing the results in a file local to the deployment machine

    This is acheived by consecutively calling get() then post() to 
    extract the information and write it to the raw_data_file

    the goal is to collect:
    - the binary blob
    - the true ISA

    params:
        - server(object): object used to communicate with praetorian server
        - num_obs(int): number of observations to request
        - data_file(string): the name of the file to store the data in
    """ 

    print("[...] collecting data")
    training_data = []

    for i in range(hyper_params.NUM_OBS):
        # make arbitrary guess to get observation and label
        server.get()
        server.post('arm')

        # convert to hex for easier analysis
        hex_blob = bytes.hex(server.binary)

        training_data.append({
            "label": server.ans,
            "blob": hex_blob,
            })

    # store it in the json file
    with open(hyper_params.RAW_DATA_FILE, 'w') as file:
        json.dump(training_data, file)
        
    print("[+] collected data")

