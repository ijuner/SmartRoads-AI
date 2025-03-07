#### Docker setup

##### **Prerequisites**
- Make sure that you download and install Docker Desktop 
  - [Download Docker Desktop for Windows](https://docs.docker.com/desktop/setup/install/windows-install/)

- You can see two docker files 
  - ```Dockerfile-backend```
  - ```Dockerfile-frontend```
- There is a ```docker-compose.yml``` file as well.
- These three files help you setup docker images.
- We will be creating two docker images one for frontend and one for backend
- In order to do this Please run the following command in terminal by moving to this folder in your terminal
  -  ```docker-compose up --build```
  - Once you run the above command it will setup two images locally. It might take some time ~10-20 minutes to do so.
  - To verify that everything is working as expected:
    - Go to ```localhost:8501``` in your browser and you should be able to see the frontend.
    - Click Upload image and process image after selecting an image with a face.
    - The backend is hosted at ```localhost:8000``` already and frontend is now ideally talking to backend api at port 8000 and getting the result for you.
- Once you verify that the app is running you need to setup a repository in docker hub in order to push the local image to the docker hub.
  - Now go to hub.docker.com
  - Select Repositories in the top tabs.
  - Select Create Repository
  - Give the repository a name and Click Create.
  - Now open a command line in your local machine
    - Run the following command ```docker tag <your-local-image-name>:<local-tag-name> <your-repository-name-including-your-username>:<tag-name>```
    - Eg. ````docker tag sprint-3-api-integration-frontend:latest niks1267/sprint-3-api-integration-frontend:latest````
    - After running this command
    - You can push your local image to your repo using the following command
    - ```docker push <your-repo-name>:<tag-name>```
    - Eg. ```docker push niks1267/sprint-3-api-integration-frontend:latest```
  - Now if you visit [Docker Hub](hub.docker.com) you should see the image in your docker hub.

> Similarly you can push the backend image also to docker hub.

- Once you push the image to docker hub you can pull the docker image in a new machine with docker desktop installed.
- In the above example I will use the following to pull the docker image to my local machine
- ```docker pull niks1267/sprint-3-api-integration-backend:latest```