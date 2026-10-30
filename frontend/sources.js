"use strict";

const chatModule = document.querySelector("#chat-module");
const sourcesModule = document.querySelector("#sources-module");
const chatModuleButton = document.querySelector("#chat-module-button");
const sourcesModuleButton = document.querySelector("#sources-module-button");

const form = document.querySelector("#sources-form");
const questionInput = document.querySelector("#sources-question-input");
const submitButton = document.querySelector("#sources-submit-button");
const feedback = document.querySelector("#sources-feedback");
const errorText = document.querySelector("#sources-error");

const resultPanel = document.querySelector("#sources-result");
const resultQuestion = document.querySelector("#sources-question");
const resultAnswer = document.querySelector("#sources-answer");
const citationsContainer = document.querySelector("#sources-citations");

let isSubmitting = false;

function showModule(moduleName) {
    const showChat = moduleName === "chat";

    chatModule.hidden = !showChat;
    sourcesModule.hidden = showChat;

    chatModuleButton.setAttribute("aria-pressed", String(showChat));
    sourcesModuleButton.setAttribute("aria-pressed", String(!showChat));
}

chatModuleButton.addEventListener("click", () => {
    showModule("chat");
});

sourcesModuleButton.addEventListener("click", () => {
    showModule("sources");
});

function setPending(pending) {
    isSubmitting = pending;
    submitButton.disabled = pending;
    questionInput.readOnly = pending;
    form.setAttribute("aria-busy", String(pending));

    submitButton.textContent = pending
        ? "Preparing your answer..."
        : "Ask the Sources";
}

function clearResult() {
    resultPanel.hidden = true;
    resultQuestion.textContent = "";
    resultAnswer.textContent = "";
    citationsContainer.replaceChildren();
}

function createCitationCard(citation) {
    if (
        !Number.isInteger(citation?.number)
        || citation.number < 1
        || typeof citation.title !== "string"
        || typeof citation.excerpt !== "string"
        || typeof citation.url !== "string"
    ) {
        throw new Error("The server returned an invalid citation.");
    }

    const url = new URL(citation.url);

    if (!["https:", "http:"].includes(url.protocol)) {
        throw new Error("The server returned an unsupported source URL.");
    }

    const card = document.createElement("article");
    card.className = "source-card";

    const title = document.createElement("h3");
    title.className = "source-title";

    const link = document.createElement("a");
    link.href = url.href;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = `[${citation.number}] ${citation.title}`;

    title.append(link);

    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = "View cited passage";

    const excerpt = document.createElement("p");
    excerpt.className = "source-excerpt";
    excerpt.textContent = citation.excerpt;

    details.append(summary, excerpt);
    card.append(title, details);

    return card;
}

function renderResult(question, data) {
    if (
        !["answered", "insufficient_evidence"].includes(data?.status)
        || typeof data.answer !== "string"
        || !data.answer.trim()
        || !Array.isArray(data.citations)
    ) {
        throw new Error("The server returned an invalid answer.");
    }

    if (
        (data.status === "answered" && data.citations.length === 0)
        || (data.status === "insufficient_evidence" && data.citations.length > 0)
    ) {
        throw new Error("The server returned inconsistent citation data.");
    }

    const citationCards = data.citations.map(createCitationCard);

    resultQuestion.textContent = question;
    resultAnswer.textContent = data.answer;
    citationsContainer.replaceChildren(...citationCards);
    resultPanel.hidden = false;

    feedback.textContent = data.status === "answered"
        ? "Answer ready. Check the cited passages."
        : "No supported answer was produced.";
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    if (isSubmitting) {
        return;
    }

    errorText.hidden = true;
    errorText.textContent = "";
    feedback.textContent = "";

    const question = questionInput.value.trim();

    if (!question) {
        errorText.textContent = "Enter a question before sending.";
        errorText.hidden = false;
        questionInput.focus();
        return;
    }

    clearResult();
    setPending(true);
    feedback.textContent = "Searching the sources and preparing your answer.";

    try {
        const response = await fetch("/ask-sources", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ question }),
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

        renderResult(question, data);
    } catch (error) {
        clearResult();
        feedback.textContent = "";

        errorText.textContent = error instanceof TypeError
            ? "Could not reach the server. Check that the application is running."
            : error.message || "Something went wrong. Please try again.";

        errorText.hidden = false;
    } finally {
        setPending(false);
    }
});