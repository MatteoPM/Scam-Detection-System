import React, { useState } from "react";
import axios from "axios";
import {
  Container,
  Typography,
  TextField,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Box,
  Divider,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from "@mui/material";

const App = () => {
  const [dialogues, setDialogues] = useState([]);
  const [label, setLabel] = useState("caller");
  const [input, setInput] = useState("");
  const [result, setResult] = useState(null);
  const [backgroundStyle, setBackgroundStyle] = useState({ background: "white" });
  const [modalOpen, setModalOpen] = useState(false);

  const modernRed = "linear-gradient(to right, #ff4e50, #f9d423)";
  const modernGreen = "linear-gradient(to right, #56ab2f, #a8e063)";

  const addDialogue = () => {
    if (!input.trim()) return;
    setDialogues([...dialogues, { label, text: input }]);
    setInput("");
    // Toggle label automatically
    setLabel(label === "caller" ? "receiver" : "caller");
  };

  const removeAllDialogues = () => {
    setDialogues([]);
    setResult(null);
    setBackgroundStyle({ background: "white" });
    setModalOpen(false);
  };

  const submitDialogues = async () => {
    const formattedDialogues = dialogues.map((d) => `${d.label}: ${d.text}\n`).join("");

    try {
      const response = await axios.post("http://127.0.0.1:4545/predict", {
        dialogue: formattedDialogues,
      });
      const prediction = response.data.prediction;
      console.log("Prediction:", prediction);
      if (prediction === "Scam") {
        setBackgroundStyle({ background: modernRed, transition: "background 1.5s ease-in-out" });
        setResult("This dialogue seems suspicious!");
      } else {
        setBackgroundStyle({ background: modernGreen, transition: "background 1.5s ease-in-out" });
        setResult("This dialogue seems safe.");
      }

      setModalOpen(true); // Open the modal to display the result
    } catch (error) {
      console.error("Error:", error);
    }
  };

  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        minHeight: "100vh",
      }}
    >
      <Box
        sx={{
          flex: 1,
          ...backgroundStyle,
          padding: "20px",
        }}
      >
        <Container
          maxWidth="sm"
          sx={{
            marginTop: "20vh",
          }}
        >
          <Typography variant="h4" gutterBottom>
            Scam Detection
          </Typography>
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel id="role-select-label">Select Role</InputLabel>
            <Select labelId="role-select-label" value={label} onChange={(e) => setLabel(e.target.value)}>
              <MenuItem value="caller">Caller</MenuItem>
              <MenuItem value="receiver">Receiver</MenuItem>
            </Select>
          </FormControl>
          <TextField
            fullWidth
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Enter dialogue"
            variant="outlined"
            sx={{ mb: 2 }}
          />
          <Box sx={{ display: "flex", gap: 2, mb: 3 }}>
            <Button variant="contained" color="primary" onClick={addDialogue}>
              Add
            </Button>
            <Button variant="outlined" color="secondary" onClick={removeAllDialogues}>
              Remove All
            </Button>
          </Box>
          <Divider />
          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
            Dialogues:
          </Typography>
          <Box sx={{ mt: 3 }}>
            <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
              {dialogues.map((d, idx) => (
                <Box
                  key={idx}
                  sx={{
                    display: "flex",
                    alignItems: "center",
                    gap: 1,
                    p: 2,
                    border: "1px solid #e0e0e0",
                    borderRadius: "8px",
                    boxShadow: "0px 2px 5px rgba(0, 0, 0, 0.1)",
                    backgroundColor: idx % 2 === 0 ? "#f9f9f9" : "#ffffff",
                  }}
                >
                  <Typography
                    variant="subtitle1"
                    sx={{
                      fontWeight: "bold",
                      color: d.label === "caller" ? "#1976d2" : "#388e3c", // Blue for caller, Green for receiver
                    }}
                  >
                    {d.label}:
                  </Typography>
                  <Typography variant="body1">{d.text}</Typography>
                </Box>
              ))}
            </Box>
          </Box>

          <Button
            variant="outlined"
            color="primary"
            onClick={submitDialogues}
            fullWidth
            sx={{
              mt: 3,
              transition: "all 0.3s ease-in-out",
              "&:hover": {
                variant: "contained",
                backgroundColor: "#2196f3",
                color: "white",
                boxShadow: "0px 4px 10px rgba(0, 0, 0, 0.2)",
              },
            }}
          >
            Submit
          </Button>
          {/* Modal for Result */}
          <Dialog open={modalOpen} onClose={() => setModalOpen(false)}>
            <DialogTitle
              sx={{
                textAlign: "center",
                fontWeight: "bold",
                backgroundColor: result === "This dialogue seems suspicious!" ? "#ffe4e4" : "#e6f7e6",
              }}
            >
              Prediction Result
            </DialogTitle>
            <DialogContent
              sx={{
                textAlign: "center",
                backgroundColor: result === "This dialogue seems suspicious!" ? "#ffe4e4" : "#e6f7e6",
                borderRadius: "8px",
                padding: "20px",
              }}
            >
              <Typography
                variant="h5"
                sx={{
                  fontWeight: "bold",
                  color: result === "This dialogue seems suspicious!" ? "#d32f2f" : "#388e3c",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: 1,
                }}
              >
                {result === "This dialogue seems suspicious!" ? "⚠️" : "✅"} {result}
              </Typography>
            </DialogContent>
            <DialogActions
              sx={{
                justifyContent: "center",
                backgroundColor: result === "This dialogue seems suspicious!" ? "#ffe4e4" : "#e6f7e6",
              }}
            >
              <Button
                onClick={() => setModalOpen(false)}
                sx={{
                  color: "white",
                  backgroundColor: "#1976d2",
                  "&:hover": { backgroundColor: "#115293" },
                }}
              >
                Close
              </Button>
              <Button
                onClick={removeAllDialogues}
                sx={{
                  color: "white",
                  backgroundColor: result === "This dialogue seems suspicious!" ? "#d32f2f" : "#388e3c",
                  "&:hover": {
                    backgroundColor: result === "This dialogue seems suspicious!" ? "#b71c1c" : "#2e7d32",
                  },
                }}
              >
                Reset
              </Button>
            </DialogActions>
          </Dialog>
        </Container>
      </Box>
      <Box
        component="footer"
        sx={{
          py: 2,
          textAlign: "center",
          background: backgroundStyle.background,
          color: result ? "white" : "black",
          transition: "all 1.5s ease-in-out",
          borderTop: "1px solid #e0e0e0",
        }}
      >
        <Typography variant="body2">SENG 550 Final Project | University of Calgary | © 2024</Typography>
        <Typography variant="body2">Jinsu Kwak, Noureldin Amer, Matteo Morrone</Typography>
      </Box>
    </Box>
  );
};

export default App;
