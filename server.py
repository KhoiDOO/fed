import flwr as fl
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Federated Learning - Clean Implementation',
                    description='Federated Learning - Clean Implementation - Server',
                    epilog='Enjoy!!! - Author: https://github.com/KhoiDOO')

    parser.add_argument('--port', type=str, default="8080", 
                        help="Server port")  
    parser.add_argument('--round', type=int, default=3, 
                        help="Server round")  

    args = parser.parse_args()
    
    fl.server.start_server(server_address=f"0.0.0.0:{args.port}", 
                           config=fl.server.ServerConfig(num_rounds=args.round))