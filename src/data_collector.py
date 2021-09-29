    

def collect (server, num_obs, data_file):
    """
    collects data by continually requesting samples from the server

    the goal is to collect:
    - the binary blob
    - the 6 possible ISAs (to narrow search space)
    - the true ISA
    This is acheived by consecutively calling get() then post() to extract the information and write it to the raw_data_file

    params:
        - observations(int): number of observations to collect
    """ 
    print("[...] collecting data")
    training_data = []

    for i in range(num_obs):
        # make arbitrary guess to get observation and label
        server.get()
        server.post('arm')

        # convert to hex for easier analysis
        hex_blob = bytes.hex(server.binary)

        training_data.append({
            "label": server.ans,
            "blob": hex_blob,
            "possible_ISAs": server.targets
            })

    # store it in the json file
    with open(data_file, 'w') as file:
        json.dump(training_data, file)
        
    print("[+] collected data")


