import streamlit as st
import requests

API_UPLOAD_URL = "http://127.0.0.1:8000/upload"

def upload_documents_tab():
    # Session States
    # Used to show the success message
    if "upload_success" not in st.session_state:
        st.session_state.upload_success = False
    # Used to reset the file uploader
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
    # Used to show the error message if upload fails
    if "upload_error" not in st.session_state:
        st.session_state.upload_error = None 

    st.markdown("""
        <div style="
            padding: 5px;
            font-size: 30px;
            color: #ED7014;
            font-weight: 500;
            margin-bottom: -10px;
        ">
            Upload Files For Analyzing
        </div>
        """, unsafe_allow_html=True)

    # Show Messages
    if st.session_state.upload_success:
        st.success("Files stored in Vector DB successfully!")
        st.session_state.upload_success = False

    if st.session_state.upload_error:
        st.error(st.session_state.upload_error)
        st.session_state.upload_error = None

    # File uploader
    uploaded_files = st.file_uploader(
        "",
        type=["pdf", "txt", "csv", "xlsx", "xls", "json", "docx"],
        accept_multiple_files=True,
        # reset the uploader when a unique key generates
        key=f"uploader_{st.session_state.uploader_key}"
    )

    if uploaded_files:
        if len(uploaded_files) == 1:
            st.success(f"{len(uploaded_files)} file uploaded successfully")
        else:
            st.success(f"{len(uploaded_files)} files uploaded successfully")

        for i, file in enumerate(uploaded_files, start=1):
            st.write(f"📄 File {i}")
            st.write(f"File Name: {file.name}")
            st.write(f"File Type: {file.type}")
            st.write(f"File Size: {round(file.size/1024,1)} KB")
            st.markdown("<hr style='margin:6px 0; border:1px solid #ED7014;'>",
                        unsafe_allow_html=True)

        # Upload Button
        if st.button("Store in Vector DB"):
            st.info("Uploading files to backend...")

            try:
                files_payload = []
                for file in uploaded_files:

                    # Prepare files for FastAPI
                    files_payload.append(("files", (file.name, file.getvalue(), file.type)))

                # Call Fast API end point
                response = requests.post(API_UPLOAD_URL, files=files_payload)
                response.raise_for_status()
                result = response.json()
                

                # Handle Result
                if result.get("status") == "success":
                    st.session_state.uploader_key += 1
                    st.session_state.upload_success = True
                    st.experimental_rerun()

                elif result.get("status") == "duplicate":
                    st.session_state.uploader_key += 1
                    st.session_state.upload_error = (
                        "A file with the same name exists in the vector database. "
                        "Please rename the file and upload again."
                    )
                    st.experimental_rerun()

                else:
                    st.session_state.upload_error = (
                        "Unknown error occurred while uploading files."
                    )
                    st.experimental_rerun()

            except requests.exceptions.RequestException as e:
                st.session_state.upload_error = f"Upload failed: {str(e)}"
                st.experimental_rerun()
