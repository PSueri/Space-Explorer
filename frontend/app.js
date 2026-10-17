"use strict";

const form = document.querySelector("#chat-form");
const messageInput = document.querySelector("#message");
const submitButton = document.querySelector("#submit-button");
const resetButton = document.querySelector("#reset-button");
const statusText = document.querySelector("#status");
const errorText = document.querySelector("#error");
const answerText = document.querySelector("#answer");
const answerPanel = document.querySelector("#answer-panel");

const MAX_HISTORY_TURNS = 2;
const MAX_HISTORY_CHARACTERS = 2000;

let history = [];
let isSubmitting = false;

function characterCount(text) {
    return Array.from(text).length;
}

function historySize() {
    return history.reduce(
        (total, turn) => total
            + characterCount(turn.question)
            + characterCount(turn.answer),
        0,
    );
}

function rememberTurn(question, answer) {
    history.push({ question, answer });

    let shortened = false;

    while (
        history.length > MAX_HISTORY_TURNS
        || historySize() > MAX_HISTORY_CHARACTERS
    ) {
        history.shift();
        shortened = true;
    }

    return shortened;
}

function setPending(pending) {
    isSubmitting = pending;
    submitButton.disabled = pending;
    resetButton.disabled = pending;
    messageInput.readOnly = pending;
    answerPanel.setAttribute("aria-busy", String(pending));
    submitButton.textContent = pending
        ? "Preparing your answer..."
        : "Ask Space Explorer";
}

function clearFeedback() {
    errorText.hidden = true;
    errorText.textContent = "";
    statusText.textContent = "";
}

function showError(message) {
    errorText.textContent = message;
    errorText.hidden = false;
}

resetButton.addEventListener("click", () => {
    if (isSubmitting) {
        return;
    }

    history = [];
    form.reset();
    clearFeedback();
    answerText.textContent = "Your answer will appear here.";
    statusText.textContent = "New conversation started.";
    messageInput.focus();
});

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    if (isSubmitting) {
        return;
    }

    clearFeedback();

    const message = messageInput.value.trim();

    if (!message) {
        showError("Enter a question before sending.");
        messageInput.focus();
        return;
    }

    setPending(true);
    statusText.textContent = "Space Explorer is preparing your answer.";
    answerText.textContent = "Waiting for a response...";

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ message, history }),
        });

        let data;

        try {
            data = await response.json();
        } catch {
            throw new Error("The server returned an unreadable response.");
        }

        if (!response.ok) {
            const detail = typeof data?.detail === "string"
                ? data.detail
                : `The request failed (HTTP ${response.status}).`;

            throw new Error(detail);
        }

        if (typeof data?.answer !== "string" || !data.answer.trim()) {
            throw new Error("The server returned an empty or invalid answer.");
        }

        const answer = data.answer.trim();
        const shortened = rememberTurn(message, answer);

        answerText.textContent = answer;
        messageInput.value = "";
        statusText.textContent = shortened
            ? "Answer ready. Older context was omitted to keep the conversation small."
            : "Answer ready.";
    } catch (error) {
        answerText.textContent = "No answer was received.";
        statusText.textContent = "";

        const message = error instanceof TypeError
            ? "Could not reach the server. Check that the application is running."
            : error.message || "Something went wrong. Please try again.";

        showError(message);
    } finally {
        setPending(false);
    }
});