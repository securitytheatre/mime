"""
mime.py: A Discord bot that uses the `ctransformers` library for message inference.
The bot responds to messages mentioning it, using a language model to infer the appropriate response.


Copyright (C) github.com/securitytheatre

This file is part of mime.

mime is free software: you can redistribute it and/or modify it under the terms of the 
GNU Affero General Public License as published by the Free Software Foundation, either 
version 3 of the License, or (at your option) any later version.

mime is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with mime. 
If not, see <https://www.gnu.org/licenses/>.
"""

import os
import logging
import re
from concurrent.futures import ProcessPoolExecutor
import dotenv
import discord
import ctransformers

# Configure logging to write to a file with INFO level
handler = logging.FileHandler(filename='mime.log', encoding='utf-8', mode='w')
discord.utils.setup_logging(handler=handler, level=logging.INFO, root=True)

# Load environment variables from a .env file
dotenv.load_dotenv()

# Configuration parameters for the language model:
# - top_k: Top K candidates to consider during sampling
# - top_p: Cumulative probability threshold for candidates
# - temperature: Temperature to control randomness in output
# - repetition_penalty: Penalty applied for repeating text
# - last_n_tokens: Number of tokens to consider for context
# - seed: Random seed for reproducibility
# - max_new_tokens: Maximum number of tokens in new output
# - batch_size: Batch size for parallel processing
# - threads: Number of threads used in inference
# - stop: Token at which to stop generating output
# - stream: Stream output instead of generating all at once
# - reset: Reset the internal state after each call
config = ctransformers.llm.Config(
    # top_k=40,
    # top_p=0.95,
    temperature=0.2,
    repetition_penalty=1.1,
    last_n_tokens=64,
    # seed=-1,
    max_new_tokens=256*10,
    # batch_size=64,
    threads=8,
    stop=None,
    stream=False,
    reset=True
)

# Initialize the language model with the specified configuration and model files
# - 'MODEL_PATH': Path to the directory containing the model
# - 'MODEL_TYPE': Type of the model to be loaded (e.g., "gpt2")
# - 'MODEL_FILE': Specific file containing the model weights
# - config: Configuration object containing parameters for the model
# - local_files_only: Indicates whether to load the model from local files only
llm = ctransformers.AutoModelForCausalLM.from_pretrained(
        os.getenv('MODEL_PATH'),
        model_type=os.getenv('MODEL_TYPE'),
        model_file=os.getenv('MODEL_FILE'),
        config=ctransformers.AutoConfig(config, os.getenv('MODEL_TYPE')),
        local_files_only=True)

# Process pool executor for parallel task execution
executor = ProcessPoolExecutor(max_workers=1)

def infer_message(content):
    """
    Generate a response using the pre-configured language model.

    :param content: The input content to process.
    :return: The inferred response from the language model.
    """
    # Construct a prompt for the language model
    template = f"Prompt:\nBelow is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{content}\n\n### Response:\n"
    return llm(template)

def write_inference_to_file(inference):
    """
    Write the result of the language model's inference to a file.

    :param inference: The inferred response from the language model.
    """
    with open("inference.md", "w", encoding="utf-8") as output:
        output.write(inference)

def process_message(content, name):
    """
    Process a message by filtering the content, inferring a response using the language model,
    and writing the response to a file.

    :param content: The original content string.
    :param name: The name to filter out from the content.
    :return: The inferred response from the language model.
    """
    content = filter_content(content, name)
    inference = infer_message(content)
    write_inference_to_file(inference)
    return inference

def filter_content(content, name):
    """
    Clean the content by removing mentions and specific characters.

    :param content: The original content string.
    :param name: The name to filter out from the content.
    :return: Filtered content string.
    """
    output = content.replace(f"{name}", "")
    output = re.sub(r'[<>&@]', '', output)
    output = output.strip()
    return output

class Mime(discord.Client):
    """
    Custom Discord client class that responds to messages mentioning the bot.
    """

    async def on_ready(self):
        """
        This method is called when the bot has successfully connected to Discord.
        It logs the application ID and the user object, providing a confirmation
        that the bot is ready to receive and process messages.
        """
        logging.info("Logged on as: %s %s", self.application_id, self.user)

    async def on_message(self, message):
        """
        This method is triggered on receiving a message in any of the channels that the bot has access to.
        It processes messages based on certain criteria:
        - Ignores messages sent by the bot itself.
        - Ignores messages without any user mentions.
        - Responds to messages that mention the bot's application_id either as the first mention or elsewhere.
        - Handles the output depending on its length, either sending it directly or as an attachment if it exceeds Discord's size limits.
        """
        logging.info("Processing Message: %s", message.clean_content)

        # Ignore messages sent by the bot or without mentions
        if message.author == self.user:
            return
        elif not message.raw_mentions:
            return
        # Handle messages that mention the bot, process content, and reply
        elif self.application_id == message.raw_mentions[0]:
            output = await self.loop.run_in_executor(executor, process_message, message.clean_content, self.application.name)
            # Check for message size limits and reply appropriately
            if len(output) > int(os.getenv('DISCORD_MESSAGE_LIMIT')):
                with open("inference.md", "rb") as inference:
                    await message.reply("Response content exceeded Discord's message size limits; see inference.md", file=discord.File(inference, filename="inference.md"))
            else:
                await message.reply(output)
        # Respond to secondary mentions
        elif self.application_id in message.raw_mentions:
            await message.reply('You called?')
        else:
            return

# Instantiate and run the Discord client
intents = discord.Intents(guilds=True, guild_messages=True, message_content=True)
Mime(max_messages=50,intents=intents).run(os.getenv('TOKEN'), reconnect=True, log_handler=handler, log_level=logging.INFO)
