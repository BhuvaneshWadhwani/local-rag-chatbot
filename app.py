"""
Gradio front end for the RAG + video-retrieval chatbot. Three panels:
  1. Chat interface with persistent conversation state
  2. System prompt / retrieved-context / token-usage inspection panel
  3. Video panel with a ranked playlist that seeks & auto-plays clips

All the model, retrieval, and prompt-building logic lives in
chatbot_backend.py, which exposes per-turn functions this UI calls.

RUN FROM THE SAME PROJECT ROOT as retrieval.py, with the `outputs/`
(WikiHow FAISS index + HowTo100M clip index/metadata) and `data/` folders
already built by WikiHow_Preprocessing.py and HowTo100M_Preprocessing.py.
"""

from pathlib import Path
import gradio as gr

from chatbot_backend import (
    SYSTEM_PROMPT,
    get_response,
    count_prompt_tokens,
    get_video_playlist,
)

# Helpers

def format_playlist_rows(playlist):
    """Turn backend clip dicts into rows for the gr.Dataframe playlist."""
    rows = []
    for i, clip in enumerate(playlist, start=1):
        rows.append([
            i,
            clip.get("step_label", ""),
            Path(str(clip["video_path"])).name,
            f'{clip["start_time"]:.1f}s - {clip["end_time"]:.1f}s',
            f'{clip["score"]:.3f}',
        ])
    return rows



# Chat turn handler (send message, get response, update state)

def handle_send(user_message, history, playlist_state):
    """
    Runs on Send-click or Enter.
    Returns: cleared input, updated history, context text, token count,
             updated playlist state, updated playlist table rows.
    """
    if not user_message or not user_message.strip():
        return "", history, gr.update(), gr.update(), playlist_state, gr.update()

    history = history or []

    response, context, articles, prompt, steps = get_response(user_message, history)
    token_count = count_prompt_tokens(prompt)

    new_history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": response},
    ]

    playlist = get_video_playlist(response, steps)
    playlist_rows = format_playlist_rows(playlist)

    return "", new_history, context, token_count, playlist, playlist_rows


def clear_conversation():
    return [], "", 0, [], []


# Video panel handler

def on_playlist_select(evt: gr.SelectData, playlist_state):
    """Selecting a playlist row: load + seek + autoplay the matching clip."""
    row_idx = evt.index[0]
    clip = playlist_state[row_idx]
    video_name = Path(str(clip["video_path"])).name
    return clip["video_path"], clip["start_time"], video_name


# JS run client-side after the Python callback above returns. Reads the
# seek target from the hidden boxes' actual rendered DOM value (see the
# note above `seek_seconds_box` for why -- passing values in as js()
# function arguments proved unreliable). Re-reads them on every polling
# attempt rather than once upfront, so it's also robust against the
# hidden boxes' own DOM update lagging slightly behind this JS running.
#
# We can't just wait for 'loadedmetadata' right away either: this JS runs
# immediately after the Python callback returns, which can race against
# Gradio actually swapping the <video> tag's src to the new file in the
# DOM. If we attached a listener before that swap happens, it would fire
# for the OLD video (or never fire at all), and playback would silently
# start from 0 with no seek applied. So we poll first, waiting until the
# element's actual current source contains the expected filename, and
# only then seek.
SEEK_JS = """
() => {
    const video = document.querySelector('#video_player video');
    if (!video) return;

    let attempts = 0;
    const maxAttempts = 60; // ~3s at 50ms intervals, then give up quietly

    const trySeek = () => {
        attempts += 1;

        const secInput = document.querySelector('#seek_seconds_value input');
        const nameInput = document.querySelector('#seek_video_name_value textarea');

        if (!secInput || !nameInput || !nameInput.value) {
            if (attempts < maxAttempts) setTimeout(trySeek, 50);
            return;
        }

        const seconds = parseFloat(secInput.value);
        const videoName = nameInput.value;
        const src = decodeURIComponent(video.currentSrc || video.src || '');

        if (!src.includes(videoName)) {
            if (attempts < maxAttempts) setTimeout(trySeek, 50);
            return;
        }

        const seekAndPlay = () => {
            video.currentTime = seconds;
            video.play();
        };

        if (video.readyState >= 1) {
            seekAndPlay();
        } else {
            video.addEventListener('loadedmetadata', seekAndPlay, { once: true });
        }
    };

    trySeek();
}
"""

# Layout

with gr.Blocks(title="RAG Chatbot with Video Retrieval") as demo:
    gr.Markdown("## RAG Chatbot with Video Retrieval")

    history_state = gr.State([])       # conversation state across turns
    playlist_state = gr.State([])      # this turn's ranked video clips

    # Hidden hand-off to client-side JS (see SEEK_JS below). We initially
    # tried passing values into js() via a chained `.then(inputs=[...])`
    # step, but Gradio's `js` parameter runs relative to `fn` in a way
    # that does NOT reliably forward `inputs` as plain function
    # arguments when fn=None -- in testing this delivered `null` for
    # both values every time. Writing the values into hidden components
    # and having JS read them directly from the DOM sidesteps that
    # entirely, reusing the same Python -> output-component update path
    # that's already proven to work (video_player's src updates
    # correctly this same way). Hidden via CSS rather than visible=False,
    # since visible=False removes the component from the DOM entirely --
    # our JS needs it to still be a real, readable DOM element.
    seek_seconds_box = gr.Number(value=0, elem_id="seek_seconds_value")
    seek_video_name_box = gr.Textbox(value="", elem_id="seek_video_name_value")

    with gr.Row():
        # ---------------- Chat column ----------------
        with gr.Column(scale=2):
            # gr.Chatbot uses the OpenAI-style "messages" format
            # ({"role": ..., "content": ...}) by default from Gradio 5 on.
            # On Gradio <5, pass `type="messages"` explicitly here.
            chatbot = gr.Chatbot(label="Chat", height=420)
            with gr.Row():
                user_input = gr.Textbox(
                    placeholder="Ask a how-to question...",
                    show_label=False,
                    scale=4,
                )
                send_btn = gr.Button("Send", scale=1, variant="primary")

        # ---------------- Inspection column ----------------
        with gr.Column(scale=1):
            gr.Markdown("#### System inspection")
            system_prompt_box = gr.Textbox(
                value=SYSTEM_PROMPT.strip(),
                label="Active system prompt",
                interactive=False,
                lines=6,
            )
            context_box = gr.Textbox(
                label="Injected / retrieved context (this turn)",
                interactive=False,
                lines=6,
            )
            token_box = gr.Number(
                value=0,
                label="Current prompt token count (system + context + history)",
                interactive=False,
            )
            clear_btn = gr.Button("Clear conversation")

    gr.Markdown("---")

    # ---------------- Video panel ----------------
    with gr.Row():
        with gr.Column(scale=1):
            video_player = gr.Video(
                label="Video",
                elem_id="video_player",
                autoplay=False,
            )
        with gr.Column(scale=1):
            gr.Markdown(
                "#### Related videos\n"
                "*One clip per how-to step, ranked by the CLIP video "
                "retriever. Selecting a row seeks to that step's moment "
                "in the source video.*"
            )
            playlist = gr.Dataframe(
                headers=["Rank", "Step", "Video file", "Time range", "Score"],
                value=[],
                interactive=False,
            )

    # ---------------- wiring ----------------
    send_btn.click(
        fn=handle_send,
        inputs=[user_input, history_state, playlist_state],
        outputs=[user_input, history_state, context_box, token_box, playlist_state, playlist],
    ).then(
        fn=lambda h: h, inputs=history_state, outputs=chatbot,
    )

    user_input.submit(
        fn=handle_send,
        inputs=[user_input, history_state, playlist_state],
        outputs=[user_input, history_state, context_box, token_box, playlist_state, playlist],
    ).then(
        fn=lambda h: h, inputs=history_state, outputs=chatbot,
    )

    clear_btn.click(
        fn=clear_conversation,
        outputs=[history_state, context_box, token_box, playlist_state, playlist],
    ).then(
        fn=lambda h: h, inputs=history_state, outputs=chatbot,
    )

    playlist.select(
        fn=on_playlist_select,
        inputs=[playlist_state],
        outputs=[video_player, seek_seconds_box, seek_video_name_box],
    ).then(
        fn=None,
        inputs=None,
        outputs=None,
        js=SEEK_JS,
    )

if __name__ == "__main__":
    # Serves video files from the project directory (needed since clip
    # video_paths point into data/howto100m_bundle/videos/). Anchored to
    # this script's own location, not the terminal's cwd.
    #
    # inbrowser=True: opens your default browser to the app automatically.
    # share=True: (opt-in, not enabled here) creates a temporary public URL
    # via Gradio's tunnel (~72h), so anyone with the link can reach your
    # locally running model over the internet. Turn on only when you
    # actually want to share it, e.g. `demo.launch(..., share=True)`.
    demo.launch(
        allowed_paths=[str(Path(__file__).resolve().parent)],
        inbrowser=True,
        # Visually hides the seek hand-off boxes (see seek_seconds_box /
        # seek_video_name_box above) while keeping them as real, readable
        # DOM elements. In Gradio 6, `css` moved here from the Blocks()
        # constructor -- passing it to Blocks() instead just emits a
        # deprecation warning and silently does nothing.
        css="#seek_seconds_value, #seek_video_name_value { display: none !important; }",
        #share=True,            # to create a public link for sharing, opt-in only when you want it
    )
