import asyncio
from client_server.game_agent_server import GameServerAgent
from agents.genetic_agent import GeneticAgent

if __name__ == "__main__":
    asyncio.run(GameServerAgent(host="0.0.0.0", port=8080, agent=GeneticAgent()).start())