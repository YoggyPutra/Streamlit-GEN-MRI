import streamlit as st
import base64
import tensorflow as tf
import cv2
from tensorflow import keras
import numpy as np
import matplotlib.cm as cm
from PIL import Image

def homepage():
    st.write("""
    ## Radiology Reporting for Cardiopulmonary Disease
    """)

    st.markdown("<p align='justify'>    Penyakit kardiopulmoner adalah gangguan yang memengaruhi jantung dan paru-paru secara bersamaan, seperti gagal jantung, penyakit paru obstruktif kronis (PPOK), dan emboli paru. Kondisi ini bisa menurunkan kemampuan tubuh untuk mengalirkan dan mendistribusikan oksigen, yang berakibat pada penurunan fungsi organ dan jaringan tubuh lainnya. Gejala penyakit kardiopulmoner sering kali meliputi sesak napas, nyeri dada, kelelahan, dan batuk yang berkepanjangan. Oleh karena itu, diagnosis yang cepat dan tepat sangat penting untuk memastikan pasien mendapatkan penanganan yang sesuai.</p>"
                "<p align='justify'>    Salah satu metode utama untuk mendiagnosis penyakit kardiopulmoner adalah diagnosis x-ray dada. Melalui x-ray, dokter dapat melihat tanda-tanda kelainan seperti pembesaran jantung (indikasi gagal jantung), penumpukan cairan di paru-paru, atau sumbatan pada saluran napas yang bisa menunjukkan PPOK. Namun, masih terdapat permasalahan dalam mendiagnosis penyakit tersebut.</p>"
                "<ol>"
                "   <li>  Dari 31 studi yang mencakup 5.863 autopsi, 8% di antaranya tergolong dalam kesalahan diagnosis Kelas I yang berpotensi mempengaruhi kelangsungan hidup pasien. Kesalahan diagnosis ini mencakup emboli paru (PE), infark miokard (MI), pneumonia, dan aspergillosis sebagai penyebab umum (Winters dkk., 2012).</li>"
                "   <li> Kesalahan diagnosis berkontribusi 6.4% dari kejadian tidak diharapkan, dimana terdapat human error berkontribusi sejumlah 96.3% (Zwaan dkk., 2010).</li>"
                "   <li> Setidaknya 1 dari 20 orang dewasa mengalami kesalahan diagnosis setiap tahun, dimana setengahnya merupakan kesalahan diagnosis fatal (Singh dkk., 2014).</li> "
                "   <li> Kesalahan diagnosis pada gagal jantung berkisar mulai dari 16.1% hingga 68.5% (Wong dkk., 2021)."
                "</ol>"
                "<p align='justify'>CRR-GenAI adalah alat bantu berbasis kecerdasan buatan yang dirancang untuk menganalisis penyakit terkait jantung dan paru-paru melalui interpretasi citra x-ray dada. Dengan algoritma canggih, CRR-GenAI mampu mendeteksi serta mengklasifikasi berbagai kelainan dan gangguan kesehatan pada organ vital tersebut secara cepat dan akurat. Alat ini membantu tenaga medis dalam proses diagnosis awal, meminimalkan risiko kesalahan interpretasi, dan mempercepat pengambilan keputusan untuk penanganan yang lebih efektif.</p>"
                , unsafe_allow_html=True)


def upload():
    st.write("""
    ## Upload and Detect
    """)
    file = st.file_uploader('Choose a image file', type='png')
    if file is not None:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        st.image(opencv_image, channels='BGR')
        st.write('You selected `%s`' % file.name)
        model = tf.keras.models.load_model(r'basic_malaria_pos_neg_v1.h5')
        pred, heatmap = apply_gradcam(model, opencv_image)
        st.write('Prediction: ' + pred)
        # super_imposed = display_gradcam(opencv_image, heatmap)
        # st.image(super_imposed)



def video():
    st.write("""
    ## Tutorial
    """)
    st.video(r'demo.mp4')

def dashboard():
    st.write("""
    ## Dashboard
    ![Cool Image](https://imgur.com/a/Qk5Z1Fe)
    """)
    # st.markdown
    


def reference():
    st.write("""
    ## Reference
    """)

    st.markdown(
                "<ol>"
                "   <li> Singh, H., Meyer, A. N. D., & Thomas, E. J. (2014). The frequency of diagnostic errors in outpatient care: estimations from three large observational studies involving US adult populations. BMJ Quality & Safety, 23(9), 727–731. https://doi.org/10.1136/bmjqs-2013-002627</li>"
                "   <li> Winters, B., Custer, J., Galvagno, S. M., Colantuoni, E., Kapoor, S. G., Lee, H., Goode, V., Robinson, K., Nakhasi, A., Pronovost, P., & Newman-Toker, D. (2012). Diagnostic errors in the intensive care unit: a systematic review of autopsy studies. BMJ Quality & Safety, 21(11), 894–902. https://doi.org/10.1136/bmjqs-2012-000803</li>"
                "   <li> Wong, C. W., Tafuro, J., Azam, Z., Satchithananda, D., Duckett, S., Barker, D., Patwala, A., Ahmed, F. Z., Mallen, C., & Kwok, C. S. (2021). Misdiagnosis of Heart Failure: A Systematic Review of the Literature. Journal of Cardiac Failure, 27(9), 925–933. https://doi.org/10.1016/j.cardfail.2021.05.014</li> "
                "   <li> Zwaan, L., Bruijne, M. de, Wagner, C., Thijs, A., Smits, M., Wal, G. van del, & Timmermans, D. R. M. (2010). Patient Record Review of the Incidence, Consequences, and Causes of Diagnostic Adverse Events. Archives of Internal Medicine, 170(12), 1015. https://doi.org/10.1001/archinternmed.2010.146"
                "</ol>"
                , unsafe_allow_html=True)
    # report_dir = r'report.pdf'
    # st_pdf_display(report_dir)

def file_selector(folder_path='.'):
    file = st.file_uploader('Choose a image file', type='png')

    if file is not None:
        file_bytes = np.asarray(bytearray(file.read()), dtype = np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        st.image(opencv_image, channels='BGR')
    return file

def st_pdf_display(pdf_file):
    with open(pdf_file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'
    st.markdown(pdf_display, unsafe_allow_html=True)

def main():
    st.title('CRR-GenAI')

    PAGES = {
        'Home': homepage,
        'Dashboard' : dashboard,
        'Upload and Classify': upload,
        'Tutorial': video,
        'Reference': reference}
    st.sidebar.title('Navigation')
    PAGES[st.sidebar.radio('Go To', ('Home', 'Dashboard', 'Upload and Classify', 'Tutorial', 'Reference'))]()
    # st.write(option_chosen)

def get_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # this converts it into RGB
    img = cv2.resize(img, (150, 150))
    img = img / 255.0
    img = img.reshape(-1, 150, 150, 3)
    return img


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.outputs]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        # st.write(pred_index)
        # st.write(preds)
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def apply_gradcam(model, image_path):
    model_copy = keras.models.clone_model(model)
    img = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)  # this converts it into RGB
    img = cv2.resize(img, (150, 150))
    
    img = img / 255.0
    img = img.reshape(-1, 150, 150, 3)

    model_copy.summary()

    model_copy.layers[-1].activation = None

    preds = model.predict(img)
    pred = np.rint(preds)
    st.write(preds)
    # heatmap = make_gradcam_heatmap(img, model, 'conv2d_3')
    heatmap = None
    if pred[0][0] == 1:
        return 'Parasitized', heatmap
    elif pred[0][1] == 1:
        return 'Uninfected', heatmap


def display_gradcam(img_path, heatmap, alpha=1):
    # Load the original image
    img = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
    img_path = Image.fromarray(img)
    img = keras.preprocessing.image.img_to_array(img_path)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    return superimposed_img

if __name__ == '__main__':
    main()