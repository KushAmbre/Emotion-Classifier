import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline, AutoTokenizer
import requests
from bs4 import BeautifulSoup

# Initialize models and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# Helper functions
def preprocess_text(text):
    lines = text.split('\n')
    return [line.strip() for line in lines if line.strip()]

def parse_and_classify_screenplay(lines, movie_title):
    elements = []
    current_scene = None
    current_character = None
    buffer_text = ""
    buffer_type = None

    for line in lines:
        if line.strip().isdigit():  # Skip page numbers
            continue
        if any(prefix in line for prefix in ('INT', 'EXT')):  # New scene
            if buffer_text:
                elements.append({'Movie': movie_title, 'Type': buffer_type, 'Scene': current_scene,
                                 'Character': current_character, 'Text': buffer_text})
                buffer_text = ""
            current_scene = line
            current_character = None
            buffer_type = 'Scene Heading'
            elements.append({'Movie': movie_title, 'Type': buffer_type, 'Scene': current_scene,
                             'Character': None, 'Text': line})
            buffer_type = None
        elif line.isupper() and not any(prefix in line for prefix in ('INT', 'EXT')):  # Character line
            if buffer_text:
                elements.append({'Movie': movie_title, 'Type': buffer_type, 'Scene': current_scene,
                                 'Character': current_character, 'Text': buffer_text})
                buffer_text = ""
            current_character = line
            buffer_type = 'Dialogue'
        else:  # Dialogue or Action Description
            element_type = 'Dialogue' if current_character else 'Action Description'
            if element_type == buffer_type or not buffer_text:
                buffer_text += (" " + line if buffer_text else line)
                buffer_type = element_type
            else:
                elements.append({'Movie': movie_title, 'Type': buffer_type, 'Scene': current_scene,
                                 'Character': current_character, 'Text': buffer_text})
                buffer_text = line
                buffer_type = element_type

    if buffer_text:
        elements.append({'Movie': movie_title, 'Type': buffer_type, 'Scene': current_scene,
                         'Character': current_character, 'Text': buffer_text})

    return elements

def assign_dialogue_numbers(df):
    df['Dialogue Number'] = None  # Initialize with None
    dialogue_indices = df[df['Type'] == 'Dialogue'].index  # Filter dialogue rows
    df.loc[dialogue_indices, 'Dialogue Number'] = range(1, len(dialogue_indices) + 1)
    return df

def perform_sentiment_analysis_and_plot_in_chunks_with_slider(df, movie_title):
    # Filter for dialogues only
    dialogue_df = df[df['Type'] == 'Dialogue'].copy()

    # Perform emotion classification
    dialogue_df['Prediction'] = dialogue_df['Text'].apply(lambda x: emotion_classifier(x, truncation=True)[0])

    # Extract emotion labels and scores into separate columns
    emotion_labels = ['joy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral']
    for emotion in emotion_labels:
        dialogue_df[emotion] = dialogue_df['Prediction'].apply(
            lambda x: x['score'] if x['label'] == emotion else 0
        )

    # Add dialogue numbers for grouping
    dialogue_df['Dialogue Number'] = range(1, len(dialogue_df) + 1)
    scene_emotion_df = dialogue_df.groupby('Dialogue Number')[emotion_labels].mean().reset_index()

    # Cache dialogue text for fast lookup
    dialogue_text_map = dict(zip(dialogue_df['Dialogue Number'], dialogue_df['Text']))

    # Prepare chunk-based plotting
    chunk_size = 100
    max_dialogue_number = scene_emotion_df['Dialogue Number'].max()

    for start in range(1, max_dialogue_number + 1, chunk_size):
        end = start + chunk_size - 1
        filtered_df = scene_emotion_df[(scene_emotion_df['Dialogue Number'] >= start) & 
                                       (scene_emotion_df['Dialogue Number'] <= end)]

        # Plot for the current chunk
        plt.figure(figsize=(14, 8))

        # Plot emotion scores
        for emotion in emotion_labels:
            if emotion == 'neutral':
                # Keep the neutral line constant at 0.5 for clarity
                plt.plot(filtered_df['Dialogue Number'], [0.5] * len(filtered_df), label='neutral', linestyle='--', color='gray')
            else:
                plt.plot(filtered_df['Dialogue Number'], filtered_df[emotion], label=emotion)

        # Add grid lines for dialogue numbers
        plt.xticks(filtered_df['Dialogue Number'], rotation=90)
        plt.grid(axis='x', linestyle='--', alpha=0.6)  # Add vertical grid lines for dialogue numbers

        # Set labels and titles
        plt.title(f"Emotion Scores for Dialogue Numbers {start} to {end}")
        plt.xlabel("Dialogue Number")
        plt.ylabel("Average Emotion Score")
        plt.legend(title="Emotions")
        plt.tight_layout()

        # Display the plot in Streamlit
        st.pyplot(plt)
        plt.close()

        # Add slider below the plot for dialogue number selection
        dialogue_num = st.slider(
            f"Select Dialogue Number for {start}-{end}",
            min_value=start, max_value=min(end, max_dialogue_number), step=1,
            key=f"slider_{start}_{end}"  # Unique key for each slider
        )

        # Display corresponding text
        selected_text = dialogue_text_map.get(dialogue_num, "No dialogue available for the selected number.")
        st.subheader(f"Selected Dialogue ({dialogue_num})")
        st.write(selected_text)


def download_script_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    script_text = soup.find("pre").get_text()
    return script_text

# Main Streamlit app
def main():
    st.title("Emotional Analytics for Movie Scripts")

    # Option 1: Upload File
    uploaded_file = st.file_uploader("Upload a screenplay (.txt)", type="txt")

    # Option 2: Provide IMSDB Script URL
    imsdb_url = st.text_input("Or enter IMSDB Script URL:")

    st.markdown("You can find movie scripts on [IMSDB](https://imsdb.com).")

    if imsdb_url:
        with st.spinner("Fetching script from IMSDB..."):
            try:
                script_text = download_script_from_url(imsdb_url)
                movie_name = imsdb_url.split("/")[-1].replace("-", " ").replace(".html", "").title()
                file_name = f"{movie_name}.txt"
                st.download_button(
                    label="Download Script as .txt",
                    data=script_text,
                    file_name=file_name,
                    mime="text/plain"
                )
                st.success(f"Script fetched successfully! Saved as '{file_name}'.")
            except Exception as e:
                st.error(f"Failed to fetch script: {e}")

    if uploaded_file is not None:
        with st.spinner("Processing the script..."):
            movie_title = os.path.splitext(uploaded_file.name)[0]
            text = uploaded_file.read().decode("utf-8")
            lines = preprocess_text(text)
            elements = parse_and_classify_screenplay(lines, movie_title)
            df = pd.DataFrame(elements)
            df = assign_dialogue_numbers(df)

            st.subheader("Parsed Screenplay Data")
            st.write(df)

            st.subheader("Emotional Arc of the Screenplay with Dialogue Selection")
            dialogue_df = perform_sentiment_analysis_and_plot_in_chunks_with_slider(df, movie_title)

            st.subheader("Additional Insights")

            if 'Character' in df.columns:
                character_counts = df['Character'].value_counts().head(10)
                st.subheader("Top 10 Characters by Number of Dialogues")
                plt.figure(figsize=(10, 6))
                character_counts.plot(kind='bar', color='skyblue', edgecolor='black')
                plt.title("Top 10 Characters by Number of Dialogues")
                plt.xlabel("Character")
                plt.ylabel("Number of Dialogues")
                plt.xticks(rotation=45)
                st.pyplot(plt)
                plt.close()

            if {'joy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral'}.issubset(dialogue_df.columns):
                emotion_sums = dialogue_df[['joy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral']].sum()
                emotion_sums_sorted = emotion_sums.sort_values(ascending=False)
                st.subheader("Top Emotions Across All Dialogues")
                plt.figure(figsize=(10, 6))
                emotion_sums_sorted.plot(kind='bar', color='orange', edgecolor='black')
                plt.title("Top Emotions Across All Dialogues")
                plt.xlabel("Emotion")
                plt.ylabel("Total Score")
                plt.xticks(rotation=45)
                st.pyplot(plt)
                plt.close()
            else:
                st.warning("Emotion analysis data is missing.")

if __name__ == "__main__":
    main()
