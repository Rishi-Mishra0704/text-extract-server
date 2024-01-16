import express from "express";
import fetch from "isomorphic-unfetch";
import pdfParse from "pdf-parse";
import { ChatOpenAI } from "@langchain/openai";
import { JsonOutputToolsParser } from "langchain/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";

const app = express();
const port = process.env.PORT;

app.use(express.json({ limit: "50mb" }));

const EXTRACTION_TEMPLATE = `Extract and save the relevant entities mentioned \
in the following passage together with their properties.

If a property is not present and is not required in the function parameters, do not include it in the output.`;

const prompt = ChatPromptTemplate.fromMessages([
  ["system", EXTRACTION_TEMPLATE],
  ["human", "{input}"],
]);

const model = new ChatOpenAI({
  modelName: "gpt-3.5-turbo-1106",
  temperature: 0,
});

const parser = new JsonOutputToolsParser();
const chain = prompt.pipe(model).pipe(parser);

app.post("/api/extract", async (req, res) => {
  try {
    const { file } = req.body;
    const dataBuffer = Buffer.from(file, "base64");
    const pdfText = await pdfParse(dataBuffer);

    const langchainResult = await chain.invoke({
      input: pdfText.text,
    });

    const openaiApiKey = process.env.OPENAI_API_KEY;

    const openaiResult = await fetch(
      "https://api.openai.com/v1/engines/davinci/completions",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${openaiApiKey}`,
        },
        body: JSON.stringify({
          prompt: langchainResult,
          max_tokens: 50,
        }),
      }
    );

    const openaiData = await openaiResult.json();

    res.status(200).json({
      langchain: langchainResult,
      openai: openaiData.choices[0]?.text,
    });
  } catch (error) {
    console.error("Error processing text:", error);
    res.status(500).json({ error: "Internal Server Error" });
  }
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
