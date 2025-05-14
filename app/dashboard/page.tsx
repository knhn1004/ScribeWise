"use client";

import { useState, useRef, DragEvent, ChangeEvent, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { 
  UploadCloud, 
  File as FileIcon, 
  Trash2, 
  Loader, 
  CheckCircle, 
  Youtube, 
  FileText,
  CornerRightUp,
  AlertCircle
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription } from "@/components/ui/alert";

// API base URL - can be moved to env variable
const API_BASE_URL = "http://localhost:8000";

// Types
interface ProcessingStatus {
  status: string;
  request_id: string;
  progress?: string;
  video_info?: any;
  error?: string;
  result?: any;
  outputs?: any;
}

interface VideoInfo {
  video_id: string;
  title: string;
  duration: number;
  url: string;
}

interface PdfInfo {
  pdf_id: string;
  filename: string;
  file_size: number;
}

// File Upload Types
interface FileWithPreview {
  id: string;
  preview: string;
  progress: number;
  name: string;
  size: number;
  type: string;
  lastModified?: number;
  file?: File;
  status?: string;
  request_id?: string;
  pdf_id?: string;
  results?: any;
  error?: string;
}

// Hook for file input
function useFileInput({ accept, maxSize }: { accept?: string; maxSize?: number }) {
  const [fileName, setFileName] = useState<string>("");
  const [error, setError] = useState<string>("");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [fileSize, setFileSize] = useState<number>(0);
  const [selectedFile, setSelectedFile] = useState<File | undefined>(undefined);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    validateAndSetFile(file);
  };

  const validateAndSetFile = (file: File | undefined) => {
    setError("");

    if (file) {
      if (maxSize && file.size > maxSize * 1024 * 1024) {
        setError(`File size must be less than ${maxSize}MB`);
        return;
      }

      if (accept && !file.type.match(accept.replace("/*", "/"))) {
        setError(`File type must be ${accept}`);
        return;
      }

      setFileSize(file.size);
      setFileName(file.name);
      setSelectedFile(file);
    }
  };

  const clearFile = () => {
    setFileName("");
    setError("");
    setFileSize(0);
    setSelectedFile(undefined);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  return {
    fileName,
    error,
    fileInputRef,
    handleFileSelect,
    validateAndSetFile,
    clearFile,
    fileSize,
    selectedFile
  };
}

// Hook for auto-resizing textarea
function useAutoResizeTextarea({
  minHeight,
  maxHeight,
}: {
  minHeight: number;
  maxHeight?: number;
}) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const adjustHeight = (reset?: boolean) => {
    const textarea = textareaRef.current;
    if (!textarea) return;

    if (reset) {
      textarea.style.height = `${minHeight}px`;
      return;
    }

    // Temporarily shrink to get the right scrollHeight
    textarea.style.height = `${minHeight}px`;

    // Calculate new height
    const newHeight = Math.max(
      minHeight,
      Math.min(
        textarea.scrollHeight,
        maxHeight ?? Number.POSITIVE_INFINITY
      )
    );

    textarea.style.height = `${newHeight}px`;
  };

  return { textareaRef, adjustHeight };
}

// File Display Component
interface FileDisplayProps {
  fileName: string;
  onClear: () => void;
}

function FileDisplay({ fileName, onClear }: FileDisplayProps) {
  return (
    <div className="flex items-center gap-2 bg-black/5 dark:bg-white/5 w-fit px-3 py-1 rounded-lg group border dark:border-white/10">
      <FileText className="w-4 h-4 text-foreground" />
      <span className="text-sm text-foreground">{fileName}</span>
      <button
        type="button"
        onClick={onClear}
        className="ml-1 p-0.5 rounded-full hover:bg-black/10 dark:hover:bg-white/10 transition-colors"
      >
        <Trash2 className="w-3 h-3 text-foreground" />
      </button>
    </div>
  );
}

// File Upload Component
function FileUpload() {
  const [files, setFiles] = useState<FileWithPreview[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  // Process dropped or selected files
  const handleFiles = (fileList: FileList) => {
    // Validate file types (only accept PDFs)
    const newFiles = Array.from(fileList)
      .filter(file => file.type === "application/pdf")
      .map((file) => ({
        id: `${URL.createObjectURL(file)}-${Date.now()}`,
        preview: URL.createObjectURL(file),
        progress: 0,
        name: file.name,
        size: file.size,
        type: file.type,
        lastModified: file.lastModified,
        file,
        status: "queued"
      }));
    
    if (newFiles.length === 0) {
      setErrorMessage("Only PDF files are supported");
      return;
    }
    
    setFiles((prev) => [...prev, ...newFiles]);
    
    // Start uploading files
    newFiles.forEach((f) => uploadFile(f));
  };

  // Process status polling
  const pollProcessingStatus = async (requestId: string, fileId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/pdf-status/${requestId}`);
      const data = await response.json();

      // Update file status
      setFiles((prev) =>
        prev.map((f) => 
          f.id === fileId 
            ? { 
                ...f, 
                status: data.status,
                progress: data.status === "complete" ? 100 : f.progress,
                results: data.status === "complete" ? data.results : f.results,
                error: data.error
              } 
            : f
        )
      );

      // Continue polling if not complete or error
      if (data.status !== "complete" && data.status !== "error") {
        setTimeout(() => pollProcessingStatus(requestId, fileId), 3000);
      }
    } catch (error) {
      setFiles((prev) =>
        prev.map((f) => 
          f.id === fileId 
            ? { ...f, status: "error", error: "Failed to check processing status" } 
            : f
        )
      );
    }
  };

  // Upload file to API
  const uploadFile = async (file: FileWithPreview) => {
    if (!file.file) return;

    try {
      // Update status to uploading
      setFiles((prev) =>
        prev.map((f) => (f.id === file.id ? { ...f, status: "uploading", progress: 10 } : f))
      );

      // Create form data
      const formData = new FormData();
      formData.append("pdf_file", file.file);

      // Upload file
      const response = await fetch(`${API_BASE_URL}/process-pdf`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      // Update file with request_id and pdf_id
      setFiles((prev) =>
        prev.map((f) =>
          f.id === file.id
            ? { 
                ...f, 
                status: "processing", 
                progress: 40,
                request_id: data.request_id,
                pdf_id: data.pdf_info.pdf_id
              }
            : f
        )
      );

      // Start polling for status
      pollProcessingStatus(data.request_id, file.id);

    } catch (error) {
      console.error("Error uploading file:", error);
      setFiles((prev) =>
        prev.map((f) =>
          f.id === file.id
            ? { ...f, status: "error", error: `Upload failed: ${error instanceof Error ? error.message : "Unknown error"}` }
            : f
        )
      );
    }
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    setErrorMessage("");
    handleFiles(e.dataTransfer.files);
  };

  const onDragOver = (e: DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = () => setIsDragging(false);

  const onSelect = (e: ChangeEvent<HTMLInputElement>) => {
    setErrorMessage("");
    if (e.target.files) handleFiles(e.target.files);
  };

  const formatFileSize = (bytes: number): string => {
    if (!bytes) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${(bytes / Math.pow(k, i)).toFixed(2)} ${sizes[i]}`;
  };

  const getStatusColor = (status: string | undefined) => {
    switch (status) {
      case "complete":
        return "bg-emerald-500";
      case "error":
        return "bg-red-500";
      case "processing":
      case "extracting_text":
        return "bg-blue-500";
      default:
        return "bg-blue-500";
    }
  };

  const getStatusText = (status: string | undefined) => {
    switch (status) {
      case "complete":
        return "Completed";
      case "error":
        return "Error";
      case "queued":
        return "Queued";
      case "uploading":
        return "Uploading";
      case "processing":
        return "Processing";
      case "extracting_text":
        return "Extracting Text";
      default:
        return "Processing";
    }
  };

  return (
    <div className="w-full max-w-3xl mx-auto">
      {errorMessage && (
        <Alert className="mb-4 bg-red-500/10 border-red-500/20 text-red-500">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{errorMessage}</AlertDescription>
        </Alert>
      )}
      
      {/* Drop zone */}
      <motion.div
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        initial={false}
        animate={{
          borderColor: isDragging ? "#3b82f6" : "#ffffff10",
          scale: isDragging ? 1.02 : 1,
        }}
        whileHover={{ scale: 1.01 }}
        transition={{ duration: 0.2 }}
        className={cn(
          "relative rounded-2xl p-8 md:p-12 text-center cursor-pointer bg-secondary/50 border border-primary/10 shadow-sm hover:shadow-md backdrop-blur group",
          isDragging && "ring-4 ring-blue-400/30 border-blue-500",
        )}
      >
        <div className="flex flex-col items-center gap-5">
          <motion.div
            animate={{ y: isDragging ? [-5, 0, -5] : 0 }}
            transition={{
              duration: 1.5,
              repeat: isDragging ? Infinity : 0,
              ease: "easeInOut",
            }}
            className="relative"
          >
            <motion.div
              animate={{
                opacity: isDragging ? [0.5, 1, 0.5] : 1,
                scale: isDragging ? [0.95, 1.05, 0.95] : 1,
              }}
              transition={{
                duration: 2,
                repeat: isDragging ? Infinity : 0,
                ease: "easeInOut",
              }}
              className="absolute -inset-4 bg-blue-400/10 rounded-full blur-md"
              style={{ display: isDragging ? "block" : "none" }}
            />
            <UploadCloud
              className={cn(
                "w-16 h-16 md:w-20 md:h-20 drop-shadow-sm",
                isDragging
                  ? "text-blue-500"
                  : "text-foreground group-hover:text-blue-500 transition-colors duration-300",
              )}
            />
          </motion.div>

          <div className="space-y-2">
            <h3 className="text-xl md:text-2xl font-semibold text-foreground">
              {isDragging
                ? "Drop files here"
                : files.length
                  ? "Add more files"
                  : "Upload PDF files"}
            </h3>
            <p className="text-muted-foreground md:text-lg max-w-md mx-auto">
              {isDragging ? (
                <span className="font-medium text-blue-500">
                  Release to upload
                </span>
              ) : (
                <>
                  Drag & drop files here, or{" "}
                  <span className="text-blue-500 font-medium">browse</span>
                </>
              )}
            </p>
            <p className="text-sm text-muted-foreground">
              Only PDF files are supported
            </p>
          </div>

          <input
            ref={inputRef}
            type="file"
            multiple
            hidden
            onChange={onSelect}
            accept="application/pdf"
          />
        </div>
      </motion.div>

      {/* Uploaded files list */}
      <div className="mt-8">
        <AnimatePresence>
          {files.length > 0 && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex justify-between items-center mb-3 px-2"
            >
              <h3 className="font-semibold text-lg md:text-xl text-foreground">
                Files ({files.length})
              </h3>
              {files.length > 1 && (
                <Button
                  variant="outline"
                  onClick={() => setFiles([])}
                  size="sm"
                >
                  Clear all
                </Button>
              )}
            </motion.div>
          )}
        </AnimatePresence>

        <div
          className={cn(
            "flex flex-col gap-3 overflow-y-auto pr-2",
            files.length > 3 && "max-h-96",
          )}
        >
          <AnimatePresence>
            {files.map((file) => (
              <motion.div
                key={file.id}
                initial={{ opacity: 0, y: 20, scale: 0.97 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -20, scale: 0.95 }}
                transition={{ type: "spring", stiffness: 300, damping: 24 }}
                className="px-4 py-4 flex items-start gap-4 rounded-xl bg-background shadow hover:shadow-md transition-all duration-200"
              >
                {/* Thumbnail */}
                <div className="relative flex-shrink-0">
                  <FileIcon className="w-16 h-16 md:w-20 md:h-20 text-muted-foreground" />
                  {file.status === "complete" && (
                    <motion.div
                      initial={{ opacity: 0, scale: 0.5 }}
                      animate={{ opacity: 1, scale: 1 }}
                      className="absolute -right-2 -bottom-2 bg-background rounded-full shadow-sm"
                    >
                      <CheckCircle className="w-5 h-5 text-emerald-500" />
                    </motion.div>
                  )}
                </div>

                {/* File info & progress */}
                <div className="flex-1 min-w-0">
                  <div className="flex flex-col gap-1 w-full">
                    {/* Filename */}
                    <div className="flex items-center gap-2 min-w-0">
                      <FileIcon className="w-5 h-5 flex-shrink-0 text-blue-500" />
                      <h4
                        className="font-medium text-base md:text-lg truncate text-foreground"
                        title={file.name}
                      >
                        {file.name}
                      </h4>
                    </div>

                    {/* Status */}
                    <div className="flex items-center justify-between gap-3 text-sm text-muted-foreground">
                      <span className="text-xs md:text-sm flex items-center gap-2">
                        <span>{formatFileSize(file.size)}</span>
                        <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-secondary">
                          {getStatusText(file.status)}
                        </span>
                      </span>
                      <span className="flex items-center gap-1.5">
                        {file.status === "error" ? (
                          <span className="text-red-500">{file.error || "Error"}</span>
                        ) : file.status !== "complete" ? (
                          <Loader className="w-4 h-4 animate-spin text-blue-500" />
                        ) : (
                          <span className="text-emerald-500">Done</span>
                        )}
                      </span>
                    </div>
                  </div>

                  {/* Progress bar */}
                  <div className="w-full h-2 bg-secondary rounded-full overflow-hidden mt-3">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${file.progress}%` }}
                      transition={{
                        duration: 0.4,
                        type: "spring",
                        stiffness: 100,
                        ease: "easeOut",
                      }}
                      className={cn(
                        "h-full rounded-full shadow-inner",
                        getStatusColor(file.status)
                      )}
                    />
                  </div>

                  {/* Results links when complete */}
                  {file.status === "complete" && file.pdf_id && (
                    <div className="mt-3 pt-3 border-t border-border">
                      <h5 className="font-medium text-sm mb-2">Results:</h5>
                      <div className="flex flex-wrap gap-2">
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => window.location.href = `/outputs/${file.pdf_id}?type=pdf`}
                        >
                          View All Outputs
                        </Button>
                      </div>
                    </div>
                  )}
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}

// YouTube URL Input Component
function YouTubeInput() {
  const [url, setUrl] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [requestId, setRequestId] = useState<string | null>(null);
  const [videoId, setVideoId] = useState<string | null>(null);
  const [processingStatus, setProcessingStatus] = useState<ProcessingStatus | null>(null);
  const [results, setResults] = useState<any | null>(null);

  // Submit YouTube URL for processing
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    // Basic YouTube URL validation
    const youtubeRegex = /^(https?:\/\/)?(www\.)?(youtube\.com|youtu\.be)\/.+$/;
    
    if (!youtubeRegex.test(url)) {
      setError("Please enter a valid YouTube URL");
      return;
    }
    
    setError("");
    setIsSubmitting(true);
    
    try {
      const response = await fetch(`${API_BASE_URL}/process`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ url }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setRequestId(data.request_id);
      setProcessingStatus({
        status: data.status,
        request_id: data.request_id,
      });

      // Start polling for status
      pollProcessingStatus(data.request_id);
    } catch (error) {
      console.error("Error submitting YouTube URL:", error);
      setError(`Failed to process video: ${error instanceof Error ? error.message : "Unknown error"}`);
      setIsSubmitting(false);
    }
  };

  // Poll for processing status
  const pollProcessingStatus = async (reqId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/status/${reqId}`);
      const data = await response.json();

      setProcessingStatus(data);
      setIsSubmitting(false);

      // Extract video_id if complete
      if (data.status === "complete") {
        if (data.result && data.result.video_info && data.result.video_info.video_id) {
          setVideoId(data.result.video_info.video_id);
          setResults(data.result);
        } else if (data.video_info && data.video_info.video_id) {
          setVideoId(data.video_info.video_id);
          // Get full results
          fetchVideoResults(data.video_info.video_id);
        }
      } else if (data.status !== "error") {
        // Continue polling if not complete
        setTimeout(() => pollProcessingStatus(reqId), 3000);
      }
    } catch (error) {
      console.error("Error checking processing status:", error);
      setError(`Failed to check processing status: ${error instanceof Error ? error.message : "Unknown error"}`);
      setIsSubmitting(false);
    }
  };

  // Fetch video results
  const fetchVideoResults = async (vidId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/videos/${vidId}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setResults(data);
    } catch (error) {
      console.error("Error fetching video results:", error);
      setError(`Failed to fetch results: ${error instanceof Error ? error.message : "Unknown error"}`);
    }
  };

  // Get progress percentage based on status
  const getProgressPercentage = () => {
    if (!processingStatus) return 0;
    
    switch (processingStatus.status) {
      case "queued":
        return 10;
      case "downloading":
        return 30;
      case "transcribing":
        return 50;
      case "summarizing":
        return 80;
      case "complete":
        return 100;
      default:
        return 0;
    }
  };

  // Reset form
  const handleReset = () => {
    setUrl("");
    setRequestId(null);
    setVideoId(null);
    setProcessingStatus(null);
    setResults(null);
    setError("");
    setIsSubmitting(false);
  };

  return (
    <div className="w-full max-w-3xl mx-auto">
      <div className="flex flex-col items-center gap-6 text-center mb-8">
        <div className="p-3 rounded-full bg-secondary/50 w-fit">
          <Youtube className="w-10 h-10 text-red-500" />
        </div>
        <div>
          <h3 className="text-xl md:text-2xl font-semibold text-foreground mb-2">
            Enter YouTube URL
          </h3>
          <p className="text-muted-foreground max-w-md">
            Paste a YouTube video URL to analyze its content
          </p>
        </div>
      </div>

      {error && (
        <Alert className="mb-4 bg-red-500/10 border-red-500/20 text-red-500">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {!processingStatus ? (
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="relative">
            <Input
              type="url"
              placeholder="https://www.youtube.com/watch?v=..."
              value={url}
              onChange={(e) => {
                setUrl(e.target.value);
                if (error) setError("");
              }}
              className={cn(
                "pr-24",
                error && "border-red-500 focus-visible:ring-red-500"
              )}
            />
            <Button 
              type="submit" 
              size="sm" 
              className="absolute right-1 top-1"
              disabled={!url || isSubmitting}
            >
              {isSubmitting ? (
                <Loader className="w-4 h-4 animate-spin" />
              ) : (
                "Analyze"
              )}
            </Button>
          </div>
        </form>
      ) : (
        <div className="space-y-4">
          {/* Processing status */}
          <Card className="p-4 bg-secondary/30">
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <h5 className="font-medium">Processing Status: {processingStatus.status}</h5>
                {processingStatus.status !== "complete" && (
                  <Loader className="w-4 h-4 animate-spin text-blue-500" />
                )}
                {processingStatus.status === "complete" && (
                  <CheckCircle className="w-4 h-4 text-emerald-500" />
                )}
              </div>
              
              <Progress value={getProgressPercentage()} className="h-2" />
              
              {processingStatus.video_info && (
                <div className="pt-2 text-sm">
                  <p className="font-medium">{processingStatus.video_info.title}</p>
                </div>
              )}
            </div>
          </Card>

          {/* Results display */}
          {processingStatus.status === "complete" && results && (
            <div className="space-y-4 mt-6">
              <div className="flex justify-between items-center">
                <h4 className="text-lg font-medium">Analysis Results</h4>
                <Button variant="outline" size="sm" onClick={handleReset}>
                  Analyze Another Video
                </Button>
              </div>

              <Card className="p-4">
                <div className="space-y-3">
                  <h5 className="font-medium text-lg">{results.video_info?.title}</h5>
                  
                  {/* Results links */}
                  <div className="flex justify-center mt-4">
                    <Button
                      onClick={() => window.location.href = `/outputs/${results.video_info?.video_id}?type=video`}
                    >
                      View All Outputs
                    </Button>
                  </div>
                </div>
              </Card>
            </div>
          )}
        </div>
      )}

      {!processingStatus && (
        <div className="mt-8 p-6 border border-dashed rounded-lg border-muted-foreground/30 text-center">
          <p className="text-muted-foreground">
            Your analyzed YouTube content will appear here
          </p>
        </div>
      )}
    </div>
  );
}

// Main Dashboard Component
function Dashboard() {
  return (
    <div className="container mx-auto py-8 px-4 max-w-6xl">
      <div className="text-center mb-10">
        <h1 className="text-3xl md:text-4xl font-bold mb-3 text-foreground">
          ScribeWise Content Analysis
        </h1>
        <p className="text-muted-foreground max-w-2xl mx-auto">
          Upload a PDF file or enter a YouTube URL to analyze content and generate summaries, notes, flashcards, and mindmaps
        </p>
      </div>

      <Card className="border-border bg-background shadow-sm">
        <CardHeader>
          <CardTitle className="text-center text-foreground">Choose Your Input Method</CardTitle>
          <CardDescription className="text-center">
            Select how you want to provide content for analysis
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="file" className="w-full">
            <TabsList className="grid w-full grid-cols-2 mb-8">
              <TabsTrigger value="file">Upload PDF</TabsTrigger>
              <TabsTrigger value="youtube">YouTube URL</TabsTrigger>
            </TabsList>
            <TabsContent value="file">
              <FileUpload />
            </TabsContent>
            <TabsContent value="youtube">
              <YouTubeInput />
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
}

export default Dashboard; 