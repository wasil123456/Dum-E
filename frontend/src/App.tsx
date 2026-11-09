import { FormEvent, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
const SESSION_STORAGE_KEY = "dume-session-id";
const AUTH_STORAGE_KEY = "dume-auth-token";

type Role = "user" | "assistant";

type Source = {
  chunk_id: string;
  source: string;
  score: number;
  rank: number;
  metadata: Record<string, unknown>;
};

type Message = {
  id: string;
  role: Role;
  content: string;
  fromCache?: boolean;
  timingsMs?: number;
  sources?: Source[];
};

type ChatResponse = {
  text: string;
  sources: Source[];
  from_cache: boolean;
  timings_ms: number;
};

type AuthResponse = {
  authenticated: boolean;
  token: string;
  session_id: string;
  email: string;
  name: string;
};

type UploadResponse = {
  filename: string;
  subject: string;
  kb_version: number;
};

type SubjectsResponse = {
  subjects: string[];
};

type DocumentInfo = {
  filename: string;
  size_bytes: number;
  uploaded_at: string;
  chunk_count: number;
};

type DocumentsResponse = {
  subject: string;
  documents: DocumentInfo[];
};

type View = "dashboard" | "chat";

function createSessionId(): string {
  if (typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `web-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function getSessionId(): string {
  const stored = window.localStorage.getItem(SESSION_STORAGE_KEY);
  if (stored) {
    return stored;
  }
  const sessionId = createSessionId();
  window.localStorage.setItem(SESSION_STORAGE_KEY, sessionId);
  return sessionId;
}

function App() {
  const [authToken, setAuthToken] = useState(
    () => window.localStorage.getItem(AUTH_STORAGE_KEY) ?? "",
  );
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [baseSessionId] = useState(getSessionId);
  const [subjects, setSubjects] = useState<string[]>([]);
  const [activeSubject, setActiveSubject] = useState("");
  const [subjectInput, setSubjectInput] = useState("");
  const [isLoadingSubjects, setIsLoadingSubjects] = useState(true);
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [isLoadingDocuments, setIsLoadingDocuments] = useState(false);
  const [view, setView] = useState<View>("dashboard");
  const [isChatActive, setIsChatActive] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState(
    "Choose a PDF to add to your knowledge base.",
  );
  const [isUploading, setIsUploading] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const sessionId = activeSubject
    ? `${baseSessionId}:${activeSubject}`
    : baseSessionId;

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isSending]);

  useEffect(() => {
    async function loadSubjects(): Promise<void> {
      try {
        if (!authToken) {
          setIsLoadingSubjects(false);
          return;
        }
        const response = await fetch(`${API_BASE_URL}/api/subjects`, {
          headers: { Authorization: `Bearer ${authToken}` },
        });
        if (!response.ok) {
          throw new Error(await readError(response, "Could not load subjects."));
        }
        const result = (await response.json()) as SubjectsResponse;
        setSubjects(result.subjects);
      } catch (subjectError) {
        setError(
          subjectError instanceof Error
            ? subjectError.message
            : "Could not load subjects.",
        );
      } finally {
        setIsLoadingSubjects(false);
      }
    }

    void loadSubjects();
  }, [authToken]);

  async function authenticatedFetch(
    input: RequestInfo | URL,
    init: RequestInit = {},
  ): Promise<Response> {
    const headers = new Headers(init.headers);
    headers.set("Authorization", `Bearer ${authToken}`);
    const response = await fetch(input, { ...init, headers });
    if (response.status === 401) {
      window.localStorage.removeItem(AUTH_STORAGE_KEY);
      setAuthToken("");
    }
    return response;
  }

  function switchSubject(subject: string): void {
    setActiveSubject(subject);
    setSelectedFile(null);
    setMessages([]);
    setError(null);
    setView("dashboard");
    setIsChatActive(false);
    setUploadStatus(
      subject
        ? `Ready to add documents to ${subject}.`
        : "Choose a subject to get started.",
    );
  }

  async function loadDocuments(subject: string): Promise<void> {
    if (!subject) {
      setDocuments([]);
      return;
    }
    setIsLoadingDocuments(true);
    try {
      const response = await authenticatedFetch(
        `${API_BASE_URL}/api/subjects/${encodeURIComponent(subject)}/files`,
      );
      if (!response.ok) {
        throw new Error(await readError(response, "Could not load documents."));
      }
      const result = (await response.json()) as DocumentsResponse;
      setDocuments(result.documents);
    } catch (documentError) {
      setError(
        documentError instanceof Error
          ? documentError.message
          : "Could not load documents.",
      );
    } finally {
      setIsLoadingDocuments(false);
    }
  }

  useEffect(() => {
    void loadDocuments(activeSubject);
  }, [activeSubject]);

  function createSubject(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    const subject = subjectInput.trim();
    if (!subject) {
      return;
    }
    if (!subjects.includes(subject)) {
      setSubjects((current) => [...current, subject].sort());
    }
    setSubjectInput("");
    switchSubject(subject);
  }

  async function uploadDocument(): Promise<void> {
    if (!selectedFile || !activeSubject) {
      return;
    }

    setIsUploading(true);
    setError(null);
    setUploadStatus("Uploading and indexing…");
    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("subject", activeSubject);

    try {
      const response = await authenticatedFetch(`${API_BASE_URL}/api/upload`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        throw new Error(await readError(response, "Upload failed."));
      }
      const result = (await response.json()) as UploadResponse;
      setUploadStatus(
        `Document indexed in ${result.subject} · KB v${result.kb_version}`,
      );
      setSelectedFile(null);
      await loadDocuments(activeSubject);
    } catch (uploadError) {
      const message =
        uploadError instanceof Error ? uploadError.message : "Upload failed.";
      setError(message);
      setUploadStatus("Upload failed. Please try again.");
    } finally {
      setIsUploading(false);
    }
  }

  async function sendQuestion(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    const cleanQuestion = question.trim();
    if (!cleanQuestion || !activeSubject || isSending) {
      return;
    }

    setQuestion("");
    setError(null);
    setMessages((current) => [
      ...current,
      { id: crypto.randomUUID(), role: "user", content: cleanQuestion },
    ]);
    setIsSending(true);

    try {
      const response = await authenticatedFetch(`${API_BASE_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          subject: activeSubject,
          question: cleanQuestion,
        }),
      });
      if (!response.ok) {
        throw new Error(await readError(response, "Chat request failed."));
      }
      const result = (await response.json()) as ChatResponse;
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: result.text,
          fromCache: result.from_cache,
          timingsMs: result.timings_ms,
          sources: result.sources,
        },
      ]);
    } catch (chatError) {
      const message =
        chatError instanceof Error ? chatError.message : "Chat request failed.";
      setError(message);
    } finally {
      setIsSending(false);
    }
  }

  async function deleteDocument(filename: string): Promise<void> {
    if (!activeSubject) return;
    try {
      const response = await authenticatedFetch(
        `${API_BASE_URL}/api/subjects/${encodeURIComponent(activeSubject)}/files/${encodeURIComponent(filename)}`,
        { method: "DELETE" },
      );
      if (!response.ok) {
        throw new Error(await readError(response, "Could not delete document."));
      }
      await loadDocuments(activeSubject);
    } catch (deleteError) {
      setError(
        deleteError instanceof Error
          ? deleteError.message
          : "Could not delete document.",
      );
    }
  }

  async function reindexDocument(filename: string): Promise<void> {
    if (!activeSubject) return;
    try {
      const response = await authenticatedFetch(
        `${API_BASE_URL}/api/subjects/${encodeURIComponent(activeSubject)}/files/${encodeURIComponent(filename)}/reindex`,
        { method: "POST" },
      );
      if (!response.ok) {
        throw new Error(await readError(response, "Could not re-index document."));
      }
      await loadDocuments(activeSubject);
    } catch (reindexError) {
      setError(
        reindexError instanceof Error
          ? reindexError.message
          : "Could not re-index document.",
      );
    }
  }

  async function resetSession(): Promise<void> {
    if (!activeSubject) {
      return;
    }
    try {
      const response = await authenticatedFetch(`${API_BASE_URL}/api/reset`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      });
      if (!response.ok) {
        throw new Error(await readError(response, "Could not reset session."));
      }
      setMessages([]);
      setError(null);
      setView("dashboard");
      setIsChatActive(false);
    } catch (resetError) {
      setError(
        resetError instanceof Error
          ? resetError.message
          : "Could not reset session.",
      );
    }
  }

  if (!authToken) {
    return <AuthScreen onAuthenticated={setAuthToken} />;
  }

  function signOut(): void {
    window.localStorage.removeItem(AUTH_STORAGE_KEY);
    setAuthToken("");
  }

  return (
    <div className={`app-shell ${isSidebarOpen ? "" : "sidebar-collapsed"}`}>
      <aside className={`sidebar ${isSidebarOpen ? "" : "sidebar-closed"}`}>
        <div className="brand">
          <div className="brand-mark">D</div>
          <div>
            <h1>DUM-E</h1>
          </div>
        </div>

        <section className="upload-card">
          <div className="section-heading">
            <span className="eyebrow">Knowledge base</span>
            <span className={`status-dot ${isUploading ? "working" : ""}`} />
          </div>
          <h2>Add a document</h2>
          <p className="muted">Upload a PDF and it will be indexed immediately.</p>
          <label className="file-picker">
            <input
              type="file"
              accept="application/pdf,.pdf"
              onChange={(event) =>
                setSelectedFile(event.target.files?.[0] ?? null)
              }
            />
            <span>{selectedFile?.name ?? "Choose PDF"}</span>
            <span className="browse">Browse</span>
          </label>
          <button
            className="primary-button full-width"
            type="button"
            disabled={!activeSubject || !selectedFile || isUploading}
            onClick={() => void uploadDocument()}
          >
            {isUploading ? "Indexing…" : "Upload & index"}
          </button>
          <p className={`upload-status ${isUploading ? "active" : ""}`}>
            {uploadStatus}
          </p>
        </section>

        <section className="subject-card">
          <div className="section-heading">
            <span className="eyebrow">Active subject</span>
            {activeSubject && <span className="active-subject-dot" />}
          </div>
          <h2>{activeSubject || "Choose a subject"}</h2>
          <select
            className="subject-select"
            value={activeSubject}
            onChange={(event) => switchSubject(event.target.value)}
            disabled={isLoadingSubjects}
            aria-label="Select subject"
          >
            <option value="">
              {isLoadingSubjects ? "Loading subjects…" : "Select existing subject"}
            </option>
            {subjects.map((subject) => (
              <option key={subject} value={subject}>
                {subject}
              </option>
            ))}
          </select>
          <form className="new-subject-form" onSubmit={createSubject}>
            <input
              value={subjectInput}
              onChange={(event) => setSubjectInput(event.target.value)}
              placeholder="New subject name"
              aria-label="New subject name"
            />
            <button type="submit" disabled={!subjectInput.trim()}>
              Add
            </button>
          </form>
          <p className="muted subject-help">
            New subjects become folders when you upload a document.
          </p>
        </section>

        <div className="sidebar-footer">
          <button className="sign-out-button" type="button" onClick={signOut}>
            <span aria-hidden="true">↪</span>
            Sign out
          </button>
        </div>
      </aside>

      <main className="chat-panel">
        <header className="chat-header w-full flex items-center justify-between px-6 py-4">
          <div>
            <button
              className="sidebar-toggle"
              type="button"
              aria-label={isSidebarOpen ? "Collapse sidebar" : "Expand sidebar"}
              aria-expanded={isSidebarOpen}
              onClick={() => setIsSidebarOpen((open) => !open)}
            >
              {isSidebarOpen ? "‹" : "☰"}
            </button>
            <span className="eyebrow">DUM-E / chat</span>
            <h2>
              {view === "dashboard" ? "Subject workspace" : "Ask your documents"}
              {activeSubject && (
                <span className="header-subject"> · {activeSubject}</span>
              )}
            </h2>
          </div>
          <div className="header-actions">
            <span className="online-indicator">● Ready</span>
            <button
              className="view-tab"
              type="button"
              onClick={() => {
                const nextView = view === "dashboard" ? "chat" : "dashboard";
                setView(nextView);
                setIsChatActive(nextView === "chat");
              }}
            >
              {view === "dashboard" ? "Chat" : "Files"}
            </button>
            <button
              className="clear-conversation-button"
              type="button"
              disabled={!activeSubject}
              onClick={() => void resetSession()}
            >
              Clear conversation
            </button>
          </div>
        </header>

        <div className="messages" aria-live="polite">
          {!activeSubject ? (
            <div className="empty-state">
              <div className="empty-icon">◈</div>
              <h3>Choose a subject to begin</h3>
              <p>Select an existing subject or create a new one in the sidebar.</p>
            </div>
          ) : view === "dashboard" ? (
            <DocumentDashboard
              subject={activeSubject}
              documents={documents}
              loading={isLoadingDocuments}
              onDelete={(filename) => void deleteDocument(filename)}
              onReindex={(filename) => void reindexDocument(filename)}
              onStartChat={() => {
                setView("chat");
                setIsChatActive(true);
              }}
            />
          ) : messages.length === 0 && !isSending ? (
            <div className="empty-state">
              <div className="empty-icon">✦</div>
              <h3>What would you like to know?</h3>
              <p>Upload a PDF, then ask a question grounded in its contents.</p>
            </div>
          ) : (
            messages.map((message) => <MessageBubble key={message.id} message={message} />)
          )}
          {isSending && (
            <div className="message-row assistant-row">
              <div className="avatar">D</div>
              <div className="typing-indicator" aria-label="DUM-E is thinking">
                <span />
                <span />
                <span />
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {error && <div className="error-banner">{error}</div>}
        {isChatActive && (
          <>
            <form className="chat-form" onSubmit={(event) => void sendQuestion(event)}>
              <input
                value={question}
                onFocus={() => {
                  setView("chat");
                  setIsChatActive(true);
                }}
                onChange={(event) => {
                  setView("chat");
                  setIsChatActive(true);
                  setQuestion(event.target.value);
                }}
                placeholder="Ask a question about your documents…"
                aria-label="Message"
                disabled={!activeSubject || isSending}
              />
              <button
                className="send-button"
                type="submit"
                aria-label="Send message"
                disabled={!activeSubject || !question.trim() || isSending}
              >
                ↑
              </button>
            </form>
            <p className="composer-note">DUM-E answers using the documents in your knowledge base.</p>
          </>
        )}
      </main>
    </div>
  );
}

function AuthScreen({
  onAuthenticated,
}: {
  onAuthenticated: (token: string) => void;
}) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    setIsSubmitting(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/api/auth/${mode}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(
          mode === "register" ? { name, email, password } : { email, password },
        ),
      });
      if (!response.ok) {
        throw new Error(await readError(response, "Authentication failed."));
      }
      const result = (await response.json()) as AuthResponse;
      window.localStorage.setItem(AUTH_STORAGE_KEY, result.token);
      onAuthenticated(result.token);
    } catch (authError) {
      setError(
        authError instanceof Error ? authError.message : "Authentication failed.",
      );
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <main className="auth-shell">
      <section className="auth-form-panel">
        <div className="auth-form-content">
          <div className="brand auth-brand">
            <div className="brand-mark">D</div>
            <div>
              <h1>DUM-E</h1>
              <p>Document assistant</p>
            </div>
          </div>
          <span className="eyebrow">{mode === "login" ? "Welcome back" : "Get started"}</span>
          <h2>{mode === "login" ? "Sign in to your workspace" : "Create your account"}</h2>
          <p className="muted">Continue your learning conversations across every subject.</p>
          <form className="auth-form" onSubmit={(event) => void submit(event)}>
            {mode === "register" && (
              <label>
                Name
                <input
                  type="text"
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  autoComplete="name"
                  maxLength={120}
                  required
                />
              </label>
            )}
            <label>
              Email
              <input
                type="email"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                autoComplete="email"
                required
              />
            </label>
            <label>
              Password
              <input
                type="password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                autoComplete={mode === "login" ? "current-password" : "new-password"}
                minLength={8}
                required
              />
            </label>
            {error && <p className="auth-error">{error}</p>}
            <button className="primary-button full-width" type="submit" disabled={isSubmitting}>
              {isSubmitting ? "Please wait…" : mode === "login" ? "Sign in" : "Create account"}
            </button>
          </form>
          <button
            className="auth-toggle"
            type="button"
            onClick={() => {
              setMode(mode === "login" ? "register" : "login");
              setError(null);
            }}
          >
            {mode === "login"
              ? "New here? Create an account"
              : "Already have an account? Sign in"}
          </button>
        </div>
      </section>
      <section className="auth-brand-panel">
        <div className="auth-brand-glow" />
        <div className="auth-brand-copy">
          <span className="eyebrow">Your knowledge, in conversation</span>
          <h1>DUM-E</h1>
          <p>your personalised learning assistant</p>
        </div>
      </section>
    </main>
  );
}

function MessageBubble({ message }: { message: Message }) {
  const [sourcesOpen, setSourcesOpen] = useState(false);

  return (
    <div className={`message-row ${message.role}-row`}>
      {message.role === "assistant" && <div className="avatar">D</div>}
      <div className="message-content">
        <div className="message-author">{message.role === "user" ? "You" : "DUM-E"}</div>
        <div className="message-bubble">
          <ReactMarkdown
            remarkPlugins={[remarkMath]}
            rehypePlugins={[rehypeKatex, rehypeHighlight]}
            components={{
              code({ className, children, ...props }) {
                const language = className?.replace("language-", "") ?? "";
                const code = String(children).replace(/\n$/, "");
                return (
                  <div className="code-block">
                    <div className="code-header">
                      <span>{language || "code"}</span>
                      <button
                        type="button"
                        onClick={() => void navigator.clipboard.writeText(code)}
                      >
                        Copy code
                      </button>
                    </div>
                    <code className={className} {...props}>
                      {children}
                    </code>
                  </div>
                );
              },
            }}
          >
            {message.content}
          </ReactMarkdown>
        </div>
        {message.role === "assistant" && (
          <div className="message-meta">
            <span className={`badge ${message.fromCache ? "cache" : "rag"}`}>
              {message.fromCache ? "Cached" : "RAG model"}
            </span>
            <span>{message.timingsMs?.toFixed(0)} ms</span>
            {message.sources && message.sources.length > 0 && (
              <button
                className="sources-toggle"
                type="button"
                onClick={() => setSourcesOpen((open) => !open)}
              >
                {sourcesOpen ? "Hide sources" : `${message.sources.length} sources`}
              </button>
            )}
          </div>
        )}
        {sourcesOpen && message.sources && (
          <div className="sources-list">
            {message.sources.map((source) => (
              <div className="source-item" key={source.chunk_id}>
                <strong>
                  {source.rank}. {source.source}
                </strong>
                <span>Relevance {source.score.toFixed(3)}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function DocumentDashboard({
  subject,
  documents,
  loading,
  onDelete,
  onReindex,
  onStartChat,
}: {
  subject: string;
  documents: DocumentInfo[];
  loading: boolean;
  onDelete: (filename: string) => void;
  onReindex: (filename: string) => void;
  onStartChat: () => void;
}) {
  return (
    <section className="document-dashboard">
      <div className="dashboard-intro">
        <span className="eyebrow">Workspace / {subject}</span>
        <h3>Your documents</h3>
        <p>Manage indexed files and keep this subject&apos;s knowledge base current.</p>
        <button className="primary-button dashboard-chat-button" type="button" onClick={onStartChat}>
          Start chatting
        </button>
      </div>
      {loading ? (
        <div className="dashboard-empty">Loading documents…</div>
      ) : documents.length === 0 ? (
        <div className="dashboard-empty">No PDFs uploaded for this subject yet.</div>
      ) : (
        <div className="document-table">
          <div className="document-row document-head">
            <span>File</span><span>Size</span><span>Uploaded</span><span>Chunks</span><span>Actions</span>
          </div>
          {documents.map((document) => (
            <div className="document-row" key={document.filename}>
              <strong>{document.filename}</strong>
              <span>{formatBytes(document.size_bytes)}</span>
              <span>{new Date(document.uploaded_at).toLocaleDateString()}</span>
              <span>{document.chunk_count.toLocaleString()}</span>
              <span className="document-actions">
                <button type="button" onClick={() => onReindex(document.filename)}>Re-index</button>
                <button type="button" className="danger-action" onClick={() => onDelete(document.filename)}>Delete</button>
              </span>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

async function readError(response: Response, fallback: string): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: string };
    return body.detail ?? fallback;
  } catch {
    return fallback;
  }
}

export default App;
