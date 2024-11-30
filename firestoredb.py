from google.cloud import firestore

def store_data(user_id, inference_id, data):
    # Inisialisasi client Firestore
    db = firestore.Client()

    # Menyimpan data ke Firestore
    doc_ref = db.collection('users') \
               .document(user_id) \
               .collection('predictions') \
               .document('type') \
               .collection('calories') \
               .document(inference_id)

    # Menyimpan data ke dokumen yang telah ditentukan
    doc_ref.set(data)
