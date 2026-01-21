from fyers_apiv3 import fyersModel
import logging
import os



class FyersClient:
    """
    A wrapper class for the Fyers API to centralize authentication
    and data request logic.
    """
    def __init__(self, credentials: dict):
        """
        Initializes the FyersClient.

        Args:
            credentials (dict): A dictionary from your config containing Fyers API
                                credentials like client_id and access_token.
        """
        self.client_id = credentials.get('client_id')
        self.access_token = credentials.get('access_token')
        log_path = os.path.join(os.getcwd())

        # The FyersModel is the actual API object from their library
        self.fyers = fyersModel.FyersModel(
            client_id=self.client_id,
            token=self.access_token,
            log_path=log_path
        )
        response = self.fyers.get_profile()
        if response.get('s') == 'ok':
            logger.info(f"Fyers client initialized successfully. Welcome, {response['data']['name']}.")
        else:
            raise ConnectionError(f"Fyers authentication failed: {response.get('message')}")

    def history(self, data: dict) -> dict:
        """
        A method to specifically request historical data.
        It wraps the underlying library call for cleaner use elsewhere.
        """
        print(f"FyersClient: Requesting history for {data.get('symbol')}")
        return self.fyers.history(data=data)