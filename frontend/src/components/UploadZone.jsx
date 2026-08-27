import { useCallback, useRef, useState } from "react";
import { UploadCloud, X, FileImage } from "lucide-react";
import { Button } from "./ui";
import "./uploadzone.css";

const ACCEPT = ["image/jpeg", "image/jpg", "image/png", "image/webp"];

export default function UploadZone({
  files,
  onFiles,
  multiple = false,
  disabled = false,
  hint = "JPG · PNG · WEBP",
}) {
  const [dragging, setDragging] = useState(false);
  const [rejected, setRejected] = useState(null);
  const inputRef = useRef(null);

  const accept = useCallback(
    (list) => {
      const arr = Array.from(list || []);
      const ok = arr.filter((f) => ACCEPT.includes(f.type));
      if (ok.length !== arr.length) {
        setRejected("Some files were skipped — only JPG, PNG and WEBP images are supported.");
      } else {
        setRejected(null);
      }
      if (ok.length) onFiles(multiple ? ok : [ok[0]]);
    },
    [multiple, onFiles]
  );

  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    if (!disabled) accept(e.dataTransfer.files);
  };

  const list = files || [];

  return (
    <div className="uz">
      {list.length === 0 ? (
        <div
          className={`uz__drop ${dragging ? "is-dragging" : ""} ${disabled ? "is-disabled" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            if (!disabled) setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => !disabled && inputRef.current?.click()}
          onKeyDown={(e) => {
            if ((e.key === "Enter" || e.key === " ") && !disabled) {
              e.preventDefault();
              inputRef.current?.click();
            }
          }}
          role="button"
          tabIndex={disabled ? -1 : 0}
          aria-label={multiple ? "Upload images" : "Upload an image"}
        >
          <span className="uz__icon">
            <UploadCloud size={26} />
          </span>
          <strong className="uz__title">
            {multiple ? "Drop images here" : "Drop image here"}
          </strong>
          <span className="uz__sub">or click to browse</span>
          <span className="uz__hint">{hint}</span>
        </div>
      ) : (
        <div className="uz__files">
          {list.map((f, i) => (
            <div className="uz__file" key={`${f.name}-${i}`}>
              <span className="uz__thumb">
                <FileImage size={16} />
              </span>
              <span className="uz__meta">
                <span className="uz__name" title={f.name}>
                  {f.name}
                </span>
                <span className="uz__size mono">{(f.size / 1024 / 1024).toFixed(2)} MB</span>
              </span>
              <button
                className="uz__remove"
                onClick={() => onFiles(list.filter((_, j) => j !== i))}
                aria-label={`Remove ${f.name}`}
                disabled={disabled}
              >
                <X size={15} />
              </button>
            </div>
          ))}
          {multiple && (
            <Button
              variant="ghost"
              size="sm"
              icon={UploadCloud}
              onClick={() => inputRef.current?.click()}
              disabled={disabled}
            >
              Add more
            </Button>
          )}
        </div>
      )}

      {rejected && <p className="uz__reject">{rejected}</p>}

      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT.join(",")}
        multiple={multiple}
        hidden
        onChange={(e) => {
          accept(e.target.files);
          e.target.value = "";
        }}
      />
    </div>
  );
}
