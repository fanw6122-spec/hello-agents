"""多智能体旅行规划系统"""

import json
from typing import Dict, Any, List
from hello_agents import SimpleAgent
from hello_agents.tools import MCPTool
from ..services.llm_service import get_llm
from ..models.schemas import TripRequest, TripPlan, DayPlan, Attraction, Meal, WeatherInfo, Location, Hotel
from ..config import get_settings
from json_repair import repair_json

# ============ Agent提示词 ============

ATTRACTION_AGENT_PROMPT = """你是景点搜索专家。你的任务是根据城市和用户偏好搜索合适的景点。

**重要提示:**
你必须使用工具来搜索景点!不要自己编造景点信息!

**工具调用格式:**
使用maps_text_search工具时,必须严格按照以下格式:
`[TOOL_CALL:amap_maps_text_search:keywords=景点关键词,city=城市名]`

**示例:**
用户: "搜索北京的历史文化景点"
你的回复: [TOOL_CALL:amap_maps_text_search:keywords=历史文化,city=北京]

用户: "搜索上海的公园"
你的回复: [TOOL_CALL:amap_maps_text_search:keywords=公园,city=上海]

**注意:**
1. 必须使用工具,不要直接回答
2. 格式必须完全正确,包括方括号和冒号
3. 参数用逗号分隔
"""

WEATHER_AGENT_PROMPT = """你是天气查询专家。你的任务是查询指定城市的天气信息。

**重要提示:**
你必须使用工具来查询天气!不要自己编造天气信息!

**工具调用格式:**
使用maps_weather工具时,必须严格按照以下格式:
`[TOOL_CALL:amap_maps_weather:city=城市名]`

**示例:**
用户: "查询北京天气"
你的回复: [TOOL_CALL:amap_maps_weather:city=北京]

用户: "上海的天气怎么样"
你的回复: [TOOL_CALL:amap_maps_weather:city=上海]

**注意:**
1. 必须使用工具,不要直接回答
2. 格式必须完全正确,包括方括号和冒号
"""

HOTEL_AGENT_PROMPT = """你是酒店推荐专家。你的任务是根据城市和景点位置推荐合适的酒店。

**重要提示:**
你必须使用工具来搜索酒店!不要自己编造酒店信息!

**工具调用格式:**
使用maps_text_search工具搜索酒店时,必须严格按照以下格式:
`[TOOL_CALL:amap_maps_text_search:keywords=酒店,city=城市名]`

**示例:**
用户: "搜索北京的酒店"
你的回复: [TOOL_CALL:amap_maps_text_search:keywords=酒店,city=北京]

**注意:**
1. 必须使用工具,不要直接回答
2. 格式必须完全正确,包括方括号和冒号
3. 关键词使用"酒店"或"宾馆"
"""


PLANNER_AGENT_PROMPT = """你是行程规划专家。你的任务是根据景点信息和天气信息,生成详细的旅行计划。

请严格按照以下JSON格式返回旅行计划:
```json
{
  "city": "城市名称",
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD",
  "days": [
    {
      "date": "YYYY-MM-DD",
      "day_index": 0,
      "description": "第1天行程概述",
      "transportation": "交通方式",
      "accommodation": "住宿类型",
      "hotel": {
        "name": "酒店名称",
        "address": "酒店地址",
        "location": {"longitude": 116.397128, "latitude": 39.916527},
        "price_range": "300-500元",
        "rating": "4.5",
        "distance": "距离景点2公里",
        "type": "经济型酒店",
        "estimated_cost": 400
      },
      "attractions": [
        {
          "name": "景点名称",
          "address": "详细地址",
          "location": {"longitude": 116.397128, "latitude": 39.916527},
          "visit_duration": 120,
          "description": "景点详细描述",
          "category": "景点类别",
          "ticket_price": 60
        }
      ],
      "meals": [
        {"type": "breakfast", "name": "早餐推荐", "description": "早餐描述", "estimated_cost": 30},
        {"type": "lunch", "name": "午餐推荐", "description": "午餐描述", "estimated_cost": 50},
        {"type": "dinner", "name": "晚餐推荐", "description": "晚餐描述", "estimated_cost": 80}
      ]
    }
  ],
  "weather_info": [
    {
      "date": "YYYY-MM-DD",
      "day_weather": "晴",
      "night_weather": "多云",
      "day_temp": 25,
      "night_temp": 15,
      "wind_direction": "南风",
      "wind_power": "1-3级"
    }
  ],
  "overall_suggestions": "总体建议",
  "budget": {
    "total_attractions": 180,
    "total_hotels": 1200,
    "total_meals": 480,
    "total_transportation": 200,
    "total": 2060
  }
}
```

**重要提示:**
1. weather_info数组必须包含每一天的天气信息
2. 温度必须是纯数字(不要带°C等单位)
3. 每天安排2-3个景点
4. 考虑景点之间的距离和游览时间
5. 每天必须包含早中晚三餐
6. 提供实用的旅行建议
7. **必须包含预算信息**:
   - 景点门票价格(ticket_price)
   - 餐饮预估费用(estimated_cost)
   - 酒店预估费用(estimated_cost)
   - 预算汇总(budget)包含各项总费用
"""


class MultiAgentTripPlanner:
    """多智能体旅行规划系统"""

    def __init__(self):
        """初始化多智能体系统"""
        print("🔄 开始初始化多智能体旅行规划系统...")

        try:
            settings = get_settings()
            self.llm = get_llm()

            # 创建共享的MCP工具(只创建一次)
            print("  - 创建共享MCP工具...")
            self.amap_tool = MCPTool(
                name="amap",
                description="高德地图服务",
                server_command=["uvx", "amap-mcp-server"],
                env={"AMAP_MAPS_API_KEY": settings.amap_api_key},
                auto_expand=True
            )
            self.amap_tool.expandable=True

            # 创建景点搜索Agent
            print("  - 创建景点搜索Agent...")
            self.attraction_agent = SimpleAgent(
                name="景点搜索专家",
                llm=self.llm,
                system_prompt=ATTRACTION_AGENT_PROMPT
            )
            self.attraction_agent.add_tool(self.amap_tool)

            # 创建天气查询Agent
            print("  - 创建天气查询Agent...")
            self.weather_agent = SimpleAgent(
                name="天气查询专家",
                llm=self.llm,
                system_prompt=WEATHER_AGENT_PROMPT
            )
            self.weather_agent.add_tool(self.amap_tool)

            # 创建酒店推荐Agent
            print("  - 创建酒店推荐Agent...")
            self.hotel_agent = SimpleAgent(
                name="酒店推荐专家",
                llm=self.llm,
                system_prompt=HOTEL_AGENT_PROMPT
            )
            self.hotel_agent.add_tool(self.amap_tool)

            # 创建行程规划Agent(不需要工具)
            print("  - 创建行程规划Agent...")
            self.planner_agent = SimpleAgent(
                name="行程规划专家",
                llm=self.llm,
                system_prompt=PLANNER_AGENT_PROMPT
            )


            print(f"✅ 多智能体系统初始化成功")
            print(f"   景点搜索Agent: {len(self.attraction_agent.list_tools())} 个工具")
            print(f"   天气查询Agent: {len(self.weather_agent.list_tools())} 个工具")
            print(f"   酒店推荐Agent: {len(self.hotel_agent.list_tools())} 个工具")

        except Exception as e:
            print(f"❌ 多智能体系统初始化失败: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def plan_trip(self, request: TripRequest) -> TripPlan:
        """
        使用多智能体协作生成旅行计划

        Args:
            request: 旅行请求

        Returns:
            旅行计划
        """
        try:
            print(f"\n{'='*60}")
            print(f"🚀 开始多智能体协作规划旅行...")
            print(f"目的地: {request.city}")
            print(f"日期: {request.start_date} 至 {request.end_date}")
            print(f"天数: {request.travel_days}天")
            print(f"偏好: {', '.join(request.preferences) if request.preferences else '无'}")
            print(f"{'='*60}\n")

            # 步骤1: 景点搜索Agent搜索景点
            print("📍 步骤1: 搜索景点...")
            attraction_query = self._build_attraction_query(request)
            attraction_response = self.attraction_agent.run(attraction_query)
            print(f"景点搜索结果: {attraction_response[:200]}...\n")

            # 步骤1.5: 查询景点真实坐标（批量模式）
            print("🗺️  步骤1.5: 查询景点真实坐标...")
            attraction_names = self._extract_attraction_names(attraction_response)
            coordinates_info = self._get_all_coordinates(attraction_names, request.city)
            print(f"   成功获取 {len(coordinates_info)} 个景点的真实坐标\n")
            
            # 步骤2: 天气查询Agent查询天气
            print("🌤️  步骤2: 查询天气...")
            weather_query = f"请查询{request.city}的天气信息"
            weather_response = self.weather_agent.run(weather_query)
            print(f"天气查询结果: {weather_response[:200]}...\n")

            # 步骤3: 酒店推荐Agent搜索酒店
            print("🏨 步骤3: 搜索酒店...")
            hotel_query = f"请搜索{request.city}的{request.accommodation}酒店"
            hotel_response = self.hotel_agent.run(hotel_query)
            # 【已改造】步骤3.5搜索景点周边真实餐厅
            print(f"酒店搜索结果: {hotel_response[:200]}...\n")

            # 步骤3.5: 【新增】搜索景点附近真实餐厅
            print("🍜 步骤3.5: 搜索景点附近真实餐厅...")
            restaurant_response = self._search_restaurants_near_attractions(
            coordinates_info, request.city)
            if restaurant_response:
                print(f"餐厅搜索结果: {restaurant_response[:200]}...\n")
            else:
                print("⚠️  未获取到餐厅信息，将使用默认餐饮推荐\n")

            if restaurant_response:
                print(f"✅ 餐厅数据已准备，长度: {len(restaurant_response)} 字符")
                print(f"餐厅数据预览: {restaurant_response[:300]}...")
            else:
                print("⚠️  无餐厅数据，LLM将自行生成餐饮推荐")
        
            # 步骤4: 行程规划Agent整合信息生成计划
            print("📋 步骤4: 生成行程计划...")
            planner_query = self._build_planner_query(request, attraction_response, weather_response, hotel_response,coordinates_info,restaurant_response)
            planner_response = self.planner_agent.run(planner_query)
            print(f"行程规划结果: {planner_response[:300]}...\n")

            # 解析最终计划
            trip_plan = self._parse_response(planner_response, request)

            print(f"{'='*60}")
            print(f"✅ 旅行计划生成完成!")
            print(f"{'='*60}\n")

            return trip_plan

        except Exception as e:
            print(f"❌ 生成旅行计划失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._create_fallback_plan(request)
    
    def _build_attraction_query(self, request: TripRequest) -> str:
        # 【已改造】景点搜索后调用geo工具获取真实坐标
        """构建景点搜索查询 - 直接包含工具调用"""
       
        keywords = []
        if request.preferences:
            # 只取第一个偏好作为关键词
            keywords = request.preferences[0]
        else:
            keywords = "景点"

        # 直接返回工具调用格式
        query = f"请使用amap_maps_text_search工具搜索{request.city}的{keywords}相关景点。\n[TOOL_CALL:amap_maps_text_search:keywords={keywords},city={request.city}]\n\n请将搜索结果按照以下格式输出，每个景点单独一行，只列出知名度高、独立售票或有明确地址的景点，不要列出公园内部的小景点名称：\n1. 景点名称\n2. 景点名称"
    
    def _extract_attraction_names(self, attraction_response: str) -> list:
        """从景点搜索Agent的响应中提取景点名称列表，兼容多种格式"""
        import re
        names = []

        # 格式1：编号列表「1. 故宫博物院」或「1、故宫博物院」
        pattern_list = r'\d+[.、]\s*\*?\*?([^\n\*（(【]+)'
        matches = re.findall(pattern_list, attraction_response)
        if matches:
            for m in matches:
                name = m.strip().rstrip('，,。.')
                name = re.sub(r'[（(].*?[）)]', '', name).strip()
                if name and 2 <= len(name) <= 20:
                    names.append(name)

        # 格式2：顿号或逗号分隔「故宫博物院、国家博物馆、...」
        if not names:
            # 找「包括」「有」后面跟着的景点列表
            pattern_inline = r'(?:包括|有|：|:)([^。\n]+)'
            inline_match = re.search(pattern_inline, attraction_response)
            if inline_match:
                raw = inline_match.group(1)
                # 按顿号、逗号、「和」分割
                parts = re.split(r'[、，,和]', raw)
                for p in parts:
                    name = p.strip().rstrip('等。.')
                    name = re.sub(r'[（(].*?[）)]', '', name).strip()
                    if name and 2 <= len(name) <= 15:
                        names.append(name)

        return names[:10]
    
    def _get_all_coordinates(self, attraction_names: list, city: str) -> dict:
        """
    直接调用高德地图API查询坐标，不经过LLM
        """
        import httpx
        from app.config import get_settings

        settings = get_settings()
        coordinates_info = {}

        for name in attraction_names[:6]:
            try:
                response = httpx.get(
                "https://restapi.amap.com/v3/geocode/geo",
                params={
                    "address": name,
                    "city": city,
                    "key": settings.amap_api_key
                },
                timeout=5
                )
                data = response.json()
                if data.get("status") == "1" and data.get("geocodes"):
                    location = data["geocodes"][0]["location"]
                    lng, lat = location.split(",")
                    coordinates_info[name] = {
                    "longitude": float(lng),
                    "latitude": float(lat)
                    }
                    print(f"    ✅ {name}: {lng}, {lat}")
            except Exception as e:
                print(f"    ❌ {name}坐标查询失败: {str(e)}")
        return coordinates_info
        

    def _search_restaurants_near_attractions(self, coordinates_info: dict, city: str) -> str:
        """
        根据景点坐标搜索附近真实餐厅

        Args:
        coordinates_info: 景点坐标字典 {景点名: {longitude, latitude}}
        city: 城市名

        Returns:
        餐厅信息文本，供行程规划Agent使用
        """
        
        import httpx
        from app.config import get_settings
        settings = get_settings()

        all_restaurant_info = []

        for attraction_name, coords in list(coordinates_info.items())[:3]:
            try:
                location = f"{coords['longitude']},{coords['latitude']}"
                response = httpx.get(
                "https://restapi.amap.com/v3/place/around",
                params={
                    "location": location,
                    "keywords": "餐厅|餐馆|饭店",
                    "radius": "1000",
                    "sortrule": "weight",
                    "offset": "5",
                    "key": settings.amap_api_key
                },
                timeout=5
                )
                data = response.json()
                if data.get("status") == "1" and data.get("pois"):
                    restaurants = data["pois"][:5]
                    names = [f"- {r['name']}（{r.get('address','地址未知')}）" 
                        for r in restaurants]
                    info = f"【{attraction_name}附近餐厅】\n" + "\n".join(names)
                    all_restaurant_info.append(info)
                    print(f"  ✅ {attraction_name}附近找到{len(restaurants)}家餐厅")
            
            except Exception as e:
                print(f"  ❌ {attraction_name}餐厅搜索失败: {str(e)}")
        return "\n\n".join(all_restaurant_info)
    
    def _get_attraction_coordinates(self, attraction_name: str, city: str) -> dict:
        """
    查询单个景点的真实坐标
    
    Args:
        attraction_name: 景点名称
        city: 城市名
        
    Returns:
        包含longitude和latitude的字典，查询失败返回None
        """
        try:
            query = f"查询{attraction_name}在{city}的坐标"
            print(f"    📍 查询坐标: {attraction_name}")
            response = self.geo_agent.run(query)
        
        # 从Agent返回的文本里提取坐标
        # 高德地图返回的坐标格式通常是 "longitude,latitude"
            import re
        # 匹配经纬度格式：数字.数字,数字.数字
            coord_pattern = r'(\d{2,3}\.\d+),(\d{2,3}\.\d+)'
            match = re.search(coord_pattern, response)
        
            if match:
                longitude = float(match.group(1))
                latitude = float(match.group(2))
                print(f"    ✅ 坐标获取成功: {longitude}, {latitude}")
                return {"longitude": longitude, "latitude": latitude}
            else:
                print(f"    ⚠️ 未能从响应中提取坐标: {response[:100]}")
                return None
            
        except Exception as e:
            print(f"    ❌ 坐标查询失败: {str(e)}")
            return None

    def _build_planner_query(self, request: TripRequest, attractions: str, weather: str, hotels: str = "",coordinates_info:dict = None,restaurant_info: str = "") -> str:
        # 【已改造】restaurant_info参数传入真实餐厅数据
        """构建行程规划查询"""
        # 构建坐标说明文字
        coords_text = ""
        if coordinates_info:
            coords_text = "\n**景点真实坐标（必须使用这些坐标，不要自己编造）:**\n"
            for name, coords in coordinates_info.items():
                coords_text += f"- {name}: 经度{coords['longitude']}, 纬度{coords['latitude']}\n"

        # 新增：构建餐厅说明文字
        restaurant_text = ""
        if restaurant_info:
            restaurant_text = f"\n**景点附近真实餐厅（餐饮安排必须从这里选择，不要编造）:**\n{restaurant_info}\n"
            
        query = f"""请根据以下信息生成{request.city}的{request.travel_days}天旅行计划:

**基本信息:**
- 城市: {request.city}
- 日期: {request.start_date} 至 {request.end_date}
- 天数: {request.travel_days}天
- 交通方式: {request.transportation}
- 住宿: {request.accommodation}
- 偏好: {', '.join(request.preferences) if request.preferences else '无'}

**景点信息:**
{attractions}
{coords_text}

**天气信息:**
{weather}

**酒店信息:**
{hotels}
{restaurant_text}
**要求:**
1. 每天安排2-3个景点
2. 每天必须包含早中晚三餐，餐厅必须从上面的真实餐厅列表中选择
3. 如果真实餐厅列表里没有早餐选项，早餐可以写「酒店附近早餐」
4. 每天推荐一个具体的酒店(从酒店信息中选择)
5. 考虑景点之间的距离和交通方式
6. 返回完整的JSON格式数据
7. 景点坐标必须使用上面提供的真实坐标，没有提供坐标的景点可以合理估算
8. JSON中所有字符串值必须写在一行内，不能包含换行符
9. 确保JSON格式完全合法，所有字符串用双引号，逗号和括号不能遗漏
"""
        if request.free_text_input:
            query += f"\n**额外要求:** {request.free_text_input}"

        return query
    
    def _parse_response(self, response: str, request: TripRequest) -> TripPlan:
        """
        解析Agent响应
        
        Args:
            response: Agent响应文本
            request: 原始请求
            
        Returns:
            旅行计划
        """
        try:
            # 尝试从响应中提取JSON
            # 查找JSON代码块
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "{" in response and "}" in response:
                # 直接查找JSON对象
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                json_str = response[json_start:json_end]
            else:
                raise ValueError("响应中未找到JSON数据")
            
            # 解析JSON，自动修复格式错误
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                print("⚠️  JSON格式有误，尝试自动修复...")
                json_str = repair_json(json_str)
                data = json.loads(json_str)
           
            # 修正overall_suggestions：如果是列表，转成字符串
            if isinstance(data.get('overall_suggestions'), list):
                data['overall_suggestions'] = '\n'.join(data['overall_suggestions'])
            
            # 修正day_index：确保从0开始
            if 'days' in data:
                for i, day in enumerate(data['days']):
                    day['day_index'] = i  # 强制从0开始

            # 转换为TripPlan对象
            trip_plan = TripPlan(**data)
            
            return trip_plan
            
        except Exception as e:
            print(f"⚠️  解析响应失败: {str(e)}")
            print(f"   将使用备用方案生成计划")
            return self._create_fallback_plan(request)
    
    def _create_fallback_plan(self, request: TripRequest) -> TripPlan:
        """创建备用计划(当Agent失败时)"""
        from datetime import datetime, timedelta
        
        # 解析日期
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        
        # 创建每日行程
        days = []
        for i in range(request.travel_days):
            current_date = start_date + timedelta(days=i)
            
            day_plan = DayPlan(
                date=current_date.strftime("%Y-%m-%d"),
                day_index=i,
                description=f"第{i+1}天行程",
                transportation=request.transportation,
                accommodation=request.accommodation,
                attractions=[
                    Attraction(
                        name=f"{request.city}景点{j+1}",
                        address=f"{request.city}市",
                        location=Location(longitude=116.4 + i*0.01 + j*0.005, latitude=39.9 + i*0.01 + j*0.005),
                        visit_duration=120,
                        description=f"这是{request.city}的著名景点",
                        category="景点"
                    )
                    for j in range(2)
                ],
                meals=[
                    Meal(type="breakfast", name=f"第{i+1}天早餐", description="当地特色早餐"),
                    Meal(type="lunch", name=f"第{i+1}天午餐", description="午餐推荐"),
                    Meal(type="dinner", name=f"第{i+1}天晚餐", description="晚餐推荐")
                ]
            )
            days.append(day_plan)
        
        return TripPlan(
            city=request.city,
            start_date=request.start_date,
            end_date=request.end_date,
            days=days,
            weather_info=[],
            overall_suggestions=f"这是为您规划的{request.city}{request.travel_days}日游行程,建议提前查看各景点的开放时间。"
        )


# 全局多智能体系统实例
_multi_agent_planner = None


def get_trip_planner_agent() -> MultiAgentTripPlanner:
    """获取多智能体旅行规划系统实例(单例模式)"""
    global _multi_agent_planner

    if _multi_agent_planner is None:
        _multi_agent_planner = MultiAgentTripPlanner()

    return _multi_agent_planner

