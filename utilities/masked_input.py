"""Masked text entry without a browser password field."""

import streamlit as st


_MASKED_INPUT = st.components.v2.component(
    "masked_text_input",
    html="""
    <label for="masked-entry"></label>
    <div class="entry">
      <input id="masked-entry" type="text" autocomplete="off"
             autocapitalize="off" spellcheck="false" class="masked">
      <button type="button" aria-label="Show value" aria-pressed="false">
        <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
          <path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12Z"/>
          <circle cx="12" cy="12" r="3"/>
          <path class="slash" d="M3 3l18 18"/>
        </svg>
      </button>
    </div>
    """,
    css="""
    label {
      display: block;
      margin-bottom: 0.5rem;
      font-size: 0.875rem;
      color: var(--st-text-color);
    }
    .entry {
      display: flex;
      border: 1px solid transparent;
      border-radius: var(--st-base-radius, 0.5rem);
      background: var(--st-secondary-background-color);
    }
    .entry:focus-within { border-color: var(--st-primary-color); }
    input {
      min-width: 0;
      width: 100%;
      padding: 0.5rem 0.75rem;
      border: 0;
      outline: none;
      background: transparent;
      color: var(--st-text-color);
      font: inherit;
    }
    /* Keep the DOM input as text even when its characters are concealed. */
    input.masked { -webkit-text-security: disc; }
    button {
      display: flex;
      align-items: center;
      padding: 0 0.75rem;
      border: 0;
      border-radius: inherit;
      background: transparent;
      color: var(--st-text-color);
      cursor: pointer;
    }
    button:focus-visible { outline: 2px solid var(--st-primary-color); }
    svg { fill: none; stroke: currentColor; stroke-width: 1.8; }
    .slash { display: none; }
    button[aria-pressed="true"] .slash { display: block; }
    """,
    js="""
    export default function ({parentElement, data, setStateValue}) {
      const input = parentElement.querySelector("input");
      const button = parentElement.querySelector("button");
      parentElement.querySelector("label").textContent = data.label;
      // Hydrate on mount, not from potentially stale data during a slow rerun.
      if (input.lastSubmittedValue === undefined) {
        input.value = data.value;
        input.lastSubmittedValue = data.value;
      }

      const submit = () => {
        if (input.value !== input.lastSubmittedValue) {
          input.lastSubmittedValue = input.value;
          setStateValue("value", input.value);
        }
      };
      input.onblur = submit;
      input.onkeydown = (event) => {
        if (event.key === "Enter" && !event.isComposing) {
          event.preventDefault();
          submit();
        }
      };
      button.onmousedown = (event) => event.preventDefault();
      button.onclick = () => {
        const masked = input.classList.toggle("masked");
        button.setAttribute("aria-label", masked ? "Show value" : "Hide value");
        button.setAttribute("aria-pressed", String(!masked));
      };
    }
    """,
)


def masked_text_input(label: str, *, key: str) -> str:
    """Return text submitted with Enter or blur, concealing it by default."""
    value = st.session_state.get(key, {}).get("value", "")
    result = _MASKED_INPUT(
        key=key,
        data={"label": label, "value": value},
        default={"value": ""},
        on_value_change=lambda: None,
    )
    return result.value
