import os
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import chain
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

from rank_llm.data import Request, Result
from rank_llm.rerank.rankllm import PromptMode, RankLLM
import re

class GroqRankLLM(RankLLM):
    def __init__(
        self,
        model_name: str,
        prompt: str,
        device: str = "cpu",
        max_tokens: Optional[int] = None,  # New parameter
    ):
        super().__init__(model="Groq", context_size=4096, prompt_mode=PromptMode.RANK_GPT, num_few_shot_examples=0)
        self.model_name = model_name
        self.device = device
        self.prompt = prompt
        self.max_tokens = max_tokens  # Store or use it as needed

        template = """{prompt}"""
        prompt_template = PromptTemplate.from_template(template)

        model = ChatGroq(temperature=0.0, model_name=self.model_name, groq_api_key=os.environ["GROQ_API_KEY"])

        output_parser = StrOutputParser()
        chain = prompt_template | model | output_parser

        self.chain = chain

    def run_llm_batched(
        self, prompts: List[Union[str, List[Dict[str, str]]]], **kwargs
    ) -> List[Tuple[str, int]]:
        responses = []
        for prompt in prompts:
            output = self.chain.invoke({"prompt": prompt})
            responses.append((output, 0))

        return responses

    def run_llm(
        self, prompt: Union[str, List[Dict[str, str]]], **kwargs
    ) -> Tuple[str, int]:
        """Abstract method to run the target language model with a passed in prompt."""
        output = self.chain.invoke({"prompt": prompt})
        return (output, 0)

    def create_prompt_batched(
        self, results: List[Result], rank_start: int, rank_end: int, batch_size: int
    ) -> List[Tuple[Union[str, List[Dict[str, str]]], int]]:
        """Abstract method to create a batch of prompts based on the results and given ranking range."""
        #This implementation creates a prompt for each individual result.
        prompts = []
        for result in results:
            prompt, _ = self.create_prompt(result, rank_start, rank_end)
            prompts.append((prompt, 0)) #Returns the created prompt, and a placeholder value of 0 for tokens (Groq doesn't expose Tokens at the moment)
        return prompts

    def create_prompt(
        self, result: Result, rank_start: int, rank_end: int
    ) -> Tuple[Union[str, List[Dict[str, str]]], int]:
        """Abstract method to create a prompt based on the result and given ranking range."""
        prompt = self.prompt.replace("{query}", result.query.text)
        passages = ""
        for i in range(rank_start, rank_end):
            passages += f"\nDocument: {result.candidates[i].doc.get('segment')}"
        prompt = prompt.replace("{documents}", passages)
        return (prompt, 0)  #Returns the created prompt, and a placeholder value of 0 for tokens (Groq doesn't expose Tokens at the moment)

    def get_num_tokens(self, prompt: Union[str, List[Dict[str, str]]]) -> int:
        """Abstract method to calculate the number of tokens contained in the given prompt."""
        return 200

    def cost_per_1k_token(self, input_token: bool) -> float:
        """Abstract method to calculate the cost per 1,000 tokens for the target language model."""
        return 0.0001

    def num_output_tokens(self) -> int:
        """Abstract method to estimate the number of tokens in the model's output."""
        return 50
    
    def rank(self, query: str, texts: List[str]) -> List[int]:
        """
        Generate a ranking for the given texts based on the query.
        This implementation constructs a prompt, runs the LLM, and parses the output to obtain ranking indices.
        """
        # Create a prompt by replacing placeholders with the actual query and documents.
        # This assumes your prompt template contains "{query}" and "{documents}".
        joined_texts = "\n".join(texts)
        prompt = self.prompt.replace("{query}", query).replace("{documents}", joined_texts)
        
        # Run the LLM with the constructed prompt.
        output, _ = self.run_llm(prompt)
        
        # Parse the output to extract ranking indices.
        # For instance, if the output is expected to be a comma-separated list of indices like "2,0,1":
        ranking = [int(num) for num in output.split(",") if num.strip().isdigit()]
        
        # Alternatively, if your output format is different, adjust the parsing logic accordingly.
        return ranking

    def rerank_batch(
        self,
        requests: List[Request],
        rank_start: int = 0,
        rank_end: int = 100,
        shuffle_candidates: bool = False,
        logging: bool = False,
        **kwargs: Any,
    ) -> List[Result]:
        """Reranks a list of requests using the RankLLM model_coordinator."""
        #This implementation uses run_llm_batched to generate the ranking
        new_requests = []

        for req in requests:
            prompt = self.prompt.replace("{query}", req.query.text)
            passages = ""
            for i in range(rank_start, rank_end):
                passages += f"\nDocument: {req.candidates[i].doc.get('segment')}"
            prompt = prompt.replace("{documents}", passages)
            new_requests.append(prompt)

        outputs = self.run_llm_batched(new_requests)
        results = []

        for idx, o in enumerate(outputs):
            request = requests[idx]
            new_candidates = []

            for score_idx, candidate in enumerate(request.candidates):
                candidate.score = int(re.search(r'\d+', o[0]).group()) # set score to the first int in the string which is the rank
                new_candidates.append(candidate)

            results.append(Result(query=request.query, candidates=new_candidates, invocations_history=[]))

        return results