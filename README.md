The project addresses the challenge of detecting cavities on grey scale radiographs, where human visual acuity is limited in discerning subtle variations. 
The model offers an objective and efficient solution for cavity detection, facilitating accelerated dental diagnoses. 
The final product utilizes a transfer learning model based on VGG16 and is designed to analyze radiographic images, providing identification of cavities and aiding dental practitioners in making informed clinical decisions swiftly. 
Users can simply upload radiographs to the webpage, where the images will then be parsed and fed into our pre-trained model, yielding the original radiograph with highlighted potential cavity areas.


Training and Validation Loss graph: 



<img width="746" alt="image" src="https://github.com/pohsuanho/CavityDetection/assets/96603996/e60f58f4-8bb4-401a-b01a-67e7d02567c5">

Final presentation: Open your command line and go to the folder where your code resides, enter command: streamlit run app.py --server.enableXsrfProtection false
A local server will pop up as below:

<img width="815" alt="image" src="https://github.com/pohsuanho/CavityDetection/assets/96603996/98b28b71-18c4-4bca-a02f-554246384139">

Upload the radiograph of your own and it will give you a processed image like this: <img width="221" alt="image" src="https://github.com/pohsuanho/CavityDetection/assets/96603996/1b44f573-9703-49a8-b57b-2946caf30b45">
circling out the concerned area.


The final accuracy on test data: 58.43%
