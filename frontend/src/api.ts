export async function detectImage(file: File) {

  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch("http://127.0.0.1:8000/detect-image", {
    method: "POST",
    body: formData
  });

  return await res.json();
}