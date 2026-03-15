const form = document.getElementById("trigger-form");
const statusEl = document.getElementById("status");
const saveStatusEl = document.getElementById("save-status");
const tagsContainer = document.getElementById("tags-container");
const addTagBtn = document.getElementById("add-tag");
const saveTagsBtn = document.getElementById("save-tags");

const setStatus = (el, message, isError = false) => {
  el.textContent = message;
  el.style.color = isError ? "#b03a2e" : "#6d5d52";
};

const clearTags = () => {
  tagsContainer.innerHTML = "";
};

const createTagChip = (value = "") => {
  const chip = document.createElement("div");
  chip.className = "tag";

  const input = document.createElement("input");
  input.type = "text";
  input.value = value;
  input.placeholder = "new tag";

  const removeBtn = document.createElement("button");
  removeBtn.type = "button";
  removeBtn.textContent = "×";
  removeBtn.addEventListener("click", () => chip.remove());

  chip.appendChild(input);
  chip.appendChild(removeBtn);
  return chip;
};

const renderTags = (tags) => {
  clearTags();
  if (!tags || tags.length === 0) {
    setStatus(saveStatusEl, "No tags yet. Add a few to get started.");
    return;
  }
  tags.forEach((tag) => tagsContainer.appendChild(createTagChip(tag)));
  setStatus(saveStatusEl, "");
};

const collectTags = () => {
  const inputs = tagsContainer.querySelectorAll("input");
  return Array.from(inputs)
    .map((input) => input.value.trim())
    .filter((value) => value.length > 0);
};

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setStatus(statusEl, "Generating tags...");

  const text = document.getElementById("trigger-text").value.trim();
  const maxTags = document.getElementById("max-tags").value;

  try {
    const response = await fetch("/extract-tags", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, max_tags: Number(maxTags) }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || "Failed to generate tags");
    }

    const data = await response.json();
    if (data.tags.length === 0) {
      alert("Your Message violated our community guidelines.");
      setStatus(statusEl, "Your Message violated our community guidelines.", true);
      return
    }
    renderTags(data.tags || []);
    setStatus(statusEl, "Tags generated. Review them below.");
  } catch (error) {
    setStatus(statusEl, error.message, true);
  }
});

addTagBtn.addEventListener("click", () => {
  tagsContainer.appendChild(createTagChip(""));
});

saveTagsBtn.addEventListener("click", async () => {
  const tags = collectTags();
  if (tags.length === 0) {
    setStatus(saveStatusEl, "Add at least one tag before saving.", true);
    return;
  }

  setStatus(saveStatusEl, "Saving tags...");

  try {
    const response = await fetch("/edit-tags", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tags }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || "Failed to save tags");
    }

    const data = await response.json();
    renderTags(data.tags || []);
    setStatus(saveStatusEl, "Tags updated and ready for the next step.");
    window.location.href = "/upload";
  } catch (error) {
    setStatus(saveStatusEl, error.message, true);
  }
});
