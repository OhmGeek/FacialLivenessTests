FROM microsoft/dotnet:sdk AS build-env
WORKDIR /LivenessTestMain/WebApplication1

COPY *.csproj ./
RUN dotnet restore

COPY . ./
run dotnet publish -c Release -o out

FROM microsoft/dotnet:aspnetcore-runtime
WORKDIR /app
COPY --from=build-env /app/out .
ENTRYPOINT ["dotnet", "aspnetapp.dll"]


