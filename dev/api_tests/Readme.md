#### Instructions to test the api endpoints and streamlit FE running

- If you don't have streamlit, flask and waitress installed.
- ```pip install waitress flask streamlit```
- Move to this folder using ```cd dev/api_tests```
- Run the waitress_server.py file in terminal using ```python waitress_server.py```
- If the run was successful the terminal would hang on Starting server on *"a local host endpoint"*
- Once the server is started run ```streamlit streamlit_tester.py```
- If successful it should redirect into your browser 
- Upload any image and give a random value into the two available fields and click done
- It should return a response 

> Note that these values are hardcoded for now just for test purpose. The idea is that as the
> api gets ready we can edit this streamlit to test it on the go