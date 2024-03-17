# Video Transcript Summarizer Tool

## Description
This Video Summarizer Tool is designed to fetch scripts from video transcriptions on Amazon Web Services and summarize them. Using LangChain, it distills the essence of educational content, making it easily accessible and understandable. This tool is perfect for educators, students, and professionals seeking quick insights from lengthy videos.

## Features
- **Automatic Transcription Fetching:** Integrates with Amazon Web Services to retrieve video transcripts.
- **Intelligent Summarization:** Uses a `LangChain` request to summarize the content, focusing on key points and lesson objectives.
- **Customizable Summaries:** Adjust the summarization criteria based on your needs, including key points, lesson objectives, and topic structure.
- **Map Reduce for Optimization:** Employs a map-reduce approach to iteratively enhance the quality of summaries.

## How It Works
1. *WIP* **Fetch Transcripts:** Connects to AWS and retrieves the transcript of the specified video.
2. **Summarization Process:** Processes the transcript using the specified `LangChain` request to extract key information based on the criteria of key points, lesson objectives, and topics.
3. **Iterative Enhancement:** Utilizes map-reduce techniques to refine summaries through multiple iterations.
