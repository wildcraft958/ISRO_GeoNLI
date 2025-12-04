
import { Upload } from "lucide-react";
import React from "react";

interface DropType {
  setimageUploaded: React.Dispatch<React.SetStateAction<boolean>> ;
    imageUploaded: boolean;
    
  setSelectedImage: React.Dispatch<React.SetStateAction<string | null>>;
    setImgFile: React.Dispatch<React.SetStateAction<File | null>>;
}

export default function DropZone({  setSelectedImage, setImgFile,setimageUploaded }: DropType) {
  const fileInputRef = React.useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = React.useState(false);

  const readFileAsDataUrl = (file: File) => {
    return new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result as string);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  };

  const handleFile = (file: File | null) => {
    if (!file) return;
    setimageUploaded(true);
    readFileAsDataUrl(file).then((data) => {
      setSelectedImage(data);
      setImgFile(file);
    });
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
    e.target.value = ""; // allow re-uploading same file
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (!isDragging) setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.currentTarget.contains(e.relatedTarget as Node)) return;
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    handleFile(file ?? null);
  };

  return (
    <div
      className={`absolute inset-0 z-50 flex items-center justify-center bg-[#000814] backdrop-blur-sm pointer-events-auto transition
        ${isDragging ? "ring-2 ring-cyan-400/60 bg-[#001429]" : ""}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <div className="text-center flex flex-col items-center gap-3">

        {/* Upload Button */}
        <div
          className="w-14 h-14 flex items-center justify-center rounded-xl bg-cyan-400/10
                      shadow-lg shadow-cyan-500/10 cursor-pointer 
                     hover:bg-cyan-400/20 transition"
          onClick={() => fileInputRef.current?.click()}
        >
          <Upload className="text-cyan-300" size={30} />
        </div>

        <p className="text-white text-lg font-semibold">Upload image</p>
        <p className="text-cyan-300/70 text-sm">Click or paste an image</p>
      </div>

      {/* Hidden file input */}
      <input
        type="file"
        accept="image/*"
        ref={fileInputRef}
        className="hidden"
        onChange={handleFileSelect}
      />
    </div>
  );
}
