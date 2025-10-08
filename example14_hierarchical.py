import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
import os
from dotenv import load_dotenv

load_dotenv()

async def main() -> str:    
    model_client=OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY")
 
    )

    support_triage= AssistantAgent(
        "support_traingle",
        model_client=model_client,
        system_message= """  You are level 1 support Triage.
        classify incoming support tickets:
        - simple/FAQ + level1_support
        - Technical  Issues + level2_support
        - complex/Engineering + level3-support
        - Account/Billing + billing_specialist
        Alwayas route to the appropriate level and provide ticket summary. """ ,
        description ="Support Triage - routes tickets to appropriate support level"
        
    )

    level1_support= AssistantAgent(
        "level1_support",
        model_client=model_client,
        system_message= """ You are level 1 support . Handle basic questions and common issues. 
        If the issue is beyond your scope, escalate to level 2 with 'ESCALATE_L2' and full context . Try to resolve
        simple password resets,
        account questions , and basic troubleshooting first. 
            """,
        description ="level 1 Support - handles basic user questions"



    )

    level2_support= AssistantAgent(
        "level2_support",
        model_client=model_client,
        system_message= """ You are level 2 Technical support . Handle complex technical issues.
        You have access to advances diagnostic tools and can perform system checks.
        If the issues\ requires engineering involvement , escalate to level 3 with 'ESCALATE_L3' .
            """,
        description ="level 2 Support - handles technical troubleshooting"

    )

    level3_support= AssistantAgent(
        "level3_support",
        model_client=model_client,
        system_message= """ You are level 3 support .Handle the most complex Technical issues.
          You can access system logs, modify configuration, and coordinate with development teams.
          Provide detailed technical analysis and permanent solutions  .
            """,
        description ="level 3 Support - handles engineering-level issues"

    )

    billing_specialist= AssistantAgent(
        "billing_specialist",
        model_client=model_client,
        system_message= """ You are a Billing Specialist. Handle all account ,
        payment, and subscription issues.
        You have access to billling systems and can process refunds,
        upgrades, and account changes.
        Escalate only if legal or executive approval is needed. 
            """,
        description ="billing specialist - handles  account and payment issues"

    )

    support_manager= AssistantAgent(
        "support_manager",
        model_client=model_client,
        system_message= """ You are a support manager. You oversee all support operations.
        You handle escalations from all levels , make policy decisions , and coordinate 
        with other departments.
        You also handle VIP customers and complex multi-department issues.  """,
        description ="support manager - supervises all support operations"

    )

    selector_model=OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY")
 
    )

    support_team= SelectorGroupChat(
        participants=[
            support_triage, level1_support, level2_support,
            level3_support,billing_specialist, support_manager
        ],
        model_client= selector_model,
        termination_condition= MaxMessageTermination(12),
        allow_repeated_speaker=True 
    )

    print("=====MULTI-LEVEL SUPPORT HIERARCY====")
    
    support_tickets=[
        """i forget my passowrd and can't log into my account you please give me 
        . can you help me reset it?""",
        #"""My API calls are returning 500 errors intermittintly
        #This started happening after yesterday's developemnt  """,
        #""" I need to upgrade my subscription but the billing page shows an error
         #Also , can I get a refund for last month's unused credits ?"""
        """Our entire production system is done. database connections are failing 
        and we're loasing revenue every minute."""
    ]

    for i , ticket in enumerate(support_tickets,1):
        print(f"\n ---support tickets #{i} ---")
        print(f"Customer Issue:{ticket}")
        print("-"*60)


        await Console(support_team.run_stream(
            task=f"New support ticket: {ticket}"
        ))

        if i < len(support_tickets):
            await support_team.reset()
            print("\n"+"="*50)

    await model_client.close()
    await selector_model.close()

asyncio.run(main())

