"use strict";

const form = document.querySelector("#chat-form");
const messageInput = document.querySelector("#message");
const submitButton = document.querySelector("#submit-button");
const statusText = document.querySelector("#status");
const errorText = document.querySelector("#error");
const answerText = document.querySelector("#answer");
const answerPanel = document.querySelector("#answer-panel");

let isSubmitting = false;

function setPending(pending) {
    isSubmitting = pending;
    submitButton.disabled = pending;
    messageInput.readOnly = pending;
    answerPanel.setAttribute("aria-busy", String(pending));
    submitButton.textContent = pending
        ? "Preparing your answer..."
        : "Ask Space Explorer";
}

function showError(message) {
    errorText.textContent = message;
    errorText.hidden = false;
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    if (isSubmitting) {
        return;
    }

    errorText.hidden = true;
    errorText.textContent = "";
    statusText.textContent = "";

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
            body: JSON.stringify({ message }),
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

        answerText.textContent = data.answer;
        statusText.textContent = "Answer ready.";
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